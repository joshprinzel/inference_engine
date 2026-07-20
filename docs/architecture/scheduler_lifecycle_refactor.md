# Scheduler Lifecycle Refactor: From Request Batching Toward Token-Work Scheduling

## Context

This project started with an Orca-style continuous batching runtime for LLM decoding. The scheduler maintained waiting requests, active slots, finished requests, and a paged KV cache manager. Once admitted, each request was fully prefetched, placed into an active slot, and then decoded one token per scheduler iteration.

That model was simple and effective for early runtime development:

```text
waiting queue
  -> admission
  -> full prefill
  -> active decode slot
  -> one decode token per step
  -> finish/free KV
```

However, as the runtime grew toward chunked prefill, scheduling policies, SLO-aware serving, and possible speculative decoding experiments, the original structure started hiding too much work inside broad methods like `init_request_state(...)` and `decode_active_requests(...)`.

The goal of this refactor was to make scheduler ownership explicit without changing runtime behavior.

---

## Original Problem

The scheduler had several concerns mixed together:

```text
admission policy
waiting queue mutation
slot assignment
KV reservation
prompt token counting
full prompt prefill
decode batch selection
decode execution
output application
finish/free logic
metrics updates
```

This worked, but it made future scheduling features harder to reason about.

The biggest architectural issue was that prefill was treated as an admission side effect. A request was admitted, immediately fully prefetched, and only then became part of the decode loop.

That blocks more advanced serving behavior such as:

```text
chunked prefill
prefill/decode interleaving
token-budget scheduling
SJF/SLO scheduling
speculative decoding
tree attention experiments
```

To support those features, the scheduler needs to think less in terms of “which requests are active?” and more in terms of “which requests receive how many tokens of work this step?”

---

## Ownership Boundary

The refactor clarified the runtime ownership model.

### Scheduler Owns Planning and Lifecycle

The scheduler owns:

```text
request queue draining
admission decisions
waiting/running/finished state
slot assignment
KV block reservation/free policy
decode batch selection
scheduled work planning
metrics and snapshots
failure/eviction handling
```

The scheduler should decide **what work should happen** and **when**.

### DecodeEngine Owns Execution

The engine owns:

```text
tokenization
prompt length counting
model-specific request initialization
prefill execution
decode execution
KV tensor materialization
CUDA/HF/backend-specific details
next-token generation
```

The engine should decide **how the selected work is computed**.

This boundary is important because future schedulers may change policy without rewriting model execution, and future engines may change execution strategy without rewriting scheduling logic.

---

## Explicit Prefill Boundary

Before the refactor, full prefill happened inside:

```python
init_request_state(request_state)
```

That method name was too vague. It hid several distinct operations:

```text
tokenize prompt
run full prompt forward pass
produce past_key_values
write KV tensors into the physical KV pool
select first next token
mark all prompt tokens computed
```

The refactor introduced an explicit engine method:

```python
prefill_request(request_state)
```

`init_request_state(...)` remains as a compatibility alias, but new scheduler code calls:

```python
self.decode_engine.prefill_request(request_state)
```

This makes the lifecycle clearer:

```text
scheduler admits request
scheduler reserves KV
engine runs prefill
scheduler places request in active slot
scheduler later selects decode work
```

Today `prefill_request(...)` still performs full prefill. Later, this boundary can evolve into chunked prefill.

---

## Admission Lifecycle Helpers

The admission path was split into named scheduler-owned helpers:

```python
reserve_kv_for_request(request_state) -> bool
prefill_admitted_request(request_state) -> None
place_request_in_slot(request_state, slot_index) -> None
```

The resulting lifecycle is:

```text
select admissions
remove request from waiting queue
find free slot
mark request admitted
reserve KV blocks
run full prefill
place request in active slot
```

This made `admit_waiting_requests()` read as orchestration rather than a large procedural block.

### Why This Matters

This creates stable seams for later changes:

```text
reserve_kv_for_request(...)
  later can support full-sequence reservation, partial reservation, or lookahead tokens

prefill_admitted_request(...)
  currently full prefill
  later chunked prefill

place_request_in_slot(...)
  currently simple slot activation
  later can track running/prefill/decode states more explicitly
```

---

## Decode Lifecycle Helpers

The decode path was similarly split into named helpers:

```python
build_runnable_decode_requests(active)
select_decode_batch(runnable)
run_decode_batch(decode_batch)
apply_decode_output(output)
```

The resulting lifecycle is:

```text
collect active requests
check KV capacity for each next token
split runnable vs stalled requests
select decode batch using policy
run decode through DecodeEngine
apply generated text/tokens
finish/free completed requests
```

This keeps the scheduler responsible for selecting runnable work while keeping model execution inside the engine.

A small behavior fix was also made: if the scheduling policy returns an empty decode batch, the scheduler now records a stall and returns instead of calling `decode_step([])`.

---

## Work Accounting Added to RequestState

Request-level work accounting was added:

```python
prefill_tokens_total
prefill_tokens_remaining
decode_tokens_total
decode_tokens_remaining
estimated_total_tokens_remaining
```

This allows the scheduler and snapshots to reason about remaining work.

Current behavior:

```text
prefill_tokens_remaining = 0 after admission
decode_tokens_remaining decreases one token per decode step
estimated_total_tokens_remaining follows remaining decode work
```

This is expected because the runtime still performs full prefill at admission time.

Future behavior with chunked prefill:

```text
prefill_tokens_remaining decreases over multiple scheduler steps
decode_tokens_remaining starts decreasing once prefill is complete
estimated_total_tokens_remaining captures both phases
```

---

## Scheduler Snapshot Improvements

The scheduler snapshot now includes aggregate work estimates:

```python
active_prefill_tokens_remaining
active_decode_tokens_remaining
active_estimated_tokens_remaining
waiting_prefill_tokens_remaining
waiting_decode_tokens_remaining
waiting_estimated_tokens_remaining
```

Each active slot also exposes per-request work fields.

This gives the runtime a control-plane view of scheduler state beyond basic counters like active requests and tokens generated.

Instead of only seeing:

```text
active = 4
waiting = 8
tokens_generated = 128
```

the runtime can now expose:

```text
active_decode_tokens_remaining = 92
active_estimated_tokens_remaining = 92
waiting_estimated_tokens_remaining = ...
```

This is the foundation for SJF, SLO-aware scheduling, and token-budget experiments.

---

## ScheduledRequestWork Abstraction

The next abstraction introduced was:

```python
@dataclass(frozen=True)
class ScheduledRequestWork:
    request_state: RequestState
    num_scheduled_tokens: int
    kind: str
```

This changes the scheduler’s internal model from:

```text
select requests for decode
```

toward:

```text
select work for requests
```

Today every selected decode request receives one scheduled token:

```python
ScheduledRequestWork(
    request_state=request,
    num_scheduled_tokens=1,
    kind="decode",
)
```

Later, the same abstraction can represent chunked prefill:

```python
ScheduledRequestWork(
    request_state=request,
    num_scheduled_tokens=32,
    kind="prefill",
)
```

Or speculative decoding:

```python
ScheduledRequestWork(
    request_state=request,
    num_scheduled_tokens=5,
    kind="spec_decode",
)
```

Or eventually tree-based verification work:

```python
ScheduledRequestWork(
    request_state=request,
    num_scheduled_tokens=7,
    kind="tree_decode",
)
```

The key shift is that the scheduler now has a place to express **how much work** is being assigned, not only which requests are selected.

---

## Relationship to Orca and vLLM

The current runtime began closer to Orca-style continuous batching:

```text
each active request contributes one decode token per iteration
requests are batched at iteration granularity
```

That remains true for current decode behavior.

However, the project is now moving toward a more general token-work scheduler inspired by systems like vLLM.

The conceptual shift is:

```text
Orca-style:
  batch active decode requests each iteration

Token-work scheduling:
  assign a token budget across requests each step
  schedule prefill, decode, or speculative work as token units
```

In vLLM-like terms, the scheduler tries to advance each request’s `num_computed_tokens` toward the total number of tokens that need to exist for that request.

Our simplified version is moving toward:

```text
num_scheduled_tokens per request
num_computed_tokens as progress
token budget per step
work kind: decode / prefill / future spec decode
```

We are not copying vLLM’s full production scheduler. We are borrowing the core abstraction: **schedule token work, not just request batches.**

---

## Why This Matters for Future Features

### Chunked Prefill

Instead of full prefill happening during admission, the scheduler can eventually schedule prefill chunks:

```text
request A -> 32 prefill tokens
request B -> 1 decode token
request C -> 16 prefill tokens
```

This enables prefill/decode interleaving and better TTFT control under long prompts.

### SJF / SLO Scheduling

Once requests expose estimated remaining work, policies can prioritize:

```text
shortest remaining work
earliest deadline
lowest TTFT risk
decode-first under latency pressure
prefill-first under throughput pressure
```

### Speculative Decoding and Tree Attention

Speculative decoding and tree verification do not fit cleanly into a “one request, one token” scheduler.

They need a representation where a request can receive multiple units of structured token work.

`ScheduledRequestWork` creates the first minimal place for that metadata to live.

### Benchmark Comparability

A passive scheduler config field was also added:

```python
max_scheduled_tokens_per_step
```

It does not yet affect behavior, but it records the future token-budget mode for experiments.

Eventually benchmark runs can compare:

```text
unbounded scheduling
8 scheduled tokens per step
32 scheduled tokens per step
128 scheduled tokens per step
```

---

## Tests Added

The refactor added tests for both lifecycle halves.

### Admission Lifecycle Test

Validates that:

```text
a waiting request is selected
KV is reserved
prefill_request(...) is called
the request is placed into an active slot
prompt_tokens and block_table are populated
metrics are not recorded prematurely
```

### Decode Lifecycle Tests

Validate that:

```text
an active request becomes runnable
decode_step(...) is called
output text is applied
tokens_generated increments
num_computed_tokens advances
the request remains active if not finished
```

Also validates that:

```text
if policy returns an empty decode batch:
  decode_stalls increments
  decode_step([]) is not called
```

### Work Plan Test

Validates that decode work planning currently assigns:

```text
1 scheduled decode token per selected request
```

This locks in today’s behavior while creating room for future prefill/spec-decode work.

---

## Interview Explanation

A strong interview summary:

```text
I started with an Orca-style continuous batching scheduler where admitted requests were fully prefetched and then decoded one token per iteration.

As I moved toward chunked prefill and SLO-aware scheduling, I realized full prefill was hidden as an admission side effect. I refactored the scheduler/engine boundary so the scheduler owns lifecycle and planning, while the DecodeEngine owns execution.

I split admission into KV reservation, explicit prefill, and slot activation. I split decode into runnable selection, policy batch selection, engine execution, and output application.

Then I added RequestState work accounting and a ScheduledRequestWork abstraction. Today it still represents one decode token per selected request, so behavior is unchanged. But architecturally the scheduler can now evolve from request batching toward token-work scheduling, where future prefill chunks, decode tokens, or speculative decode work are all scheduled through the same conceptual path.
```

The key design point:

```text
Scheduler decides what work should happen.
Engine decides how that work is executed.
```

---

## Production Code Reading

Relevant production systems to compare against:

```text
vLLM scheduler:
  token budget
  num_scheduled_tokens
  num_computed_tokens
  waiting/running request scheduling
  KV allocation before execution

Hugging Face TGI:
  continuous batching lifecycle
  router/scheduler separation
  request admission and batching

SGLang:
  serving scheduler design
  prefill/decode interleaving
  radix/prefix-cache-aware scheduling

TensorRT-LLM:
  in-flight batching
  KV cache management
  CUDA graph / execution-oriented serving abstractions
```

For vLLM specifically, read for:

```text
how running requests are traversed
how token budget is consumed
how KV slots are allocated before compute
how num_scheduled_tokens is produced
how prefill and decode are unified as token progress
```

Ignore initially:

```text
LoRA
remote KV transfer
encoder inputs
pipeline parallelism
spec decode details
Mamba-specific paths
distributed serving
```

The point is not to copy the production scheduler. The point is to understand the abstraction pressure that led production systems toward token-work scheduling.

---

## Current Architecture Checkpoint

The runtime now has:

```text
explicit prefill boundary
admission lifecycle helpers
decode lifecycle helpers
work accounting on RequestState
aggregate scheduler work snapshots
passive token budget config
ScheduledRequestWork abstraction
tests for admission, decode, and work-plan behavior
```

This is the right checkpoint before implementing actual token-budget behavior or chunked prefill.
