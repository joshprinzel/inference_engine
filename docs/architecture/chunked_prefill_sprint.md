# Chunked Prefill Sprint Note

## Sprint Goal

The goal of this sprint is to move the inference scheduler from request-level prefill execution toward token-budgeted chunked prefill.

The immediate objective is not just to make the code cleaner. The goal is to make the runtime behave more like a real LLM serving system:

* admission should not imply immediate execution
* prefill should be schedulable work
* decode should only run after prefill is complete
* long prompts should not monopolize the scheduler
* prefill progress should be visible through request state and scheduler snapshots

This sprint is the bridge between a continuous batching decode engine and a more realistic serving scheduler that can interleave prefill and decode work.

## Previous Behavior

Before this refactor, admission and prefill were coupled.

A request entered the scheduler, received KV blocks, immediately ran full prompt prefill, and then entered decode. Conceptually, the lifecycle was compressed:

```text
waiting -> full prefill during admission -> decoding
```

This worked for basic continuous batching, but it was not a good model for production-style serving. A long prompt could execute as one large prefill operation before the scheduler had another chance to make a decision.

That made it difficult to model:

* chunked prefill
* prefill/decode interleaving
* token budgets
* time-to-first-token improvements under mixed workloads
* fairness between long-prompt and short-prompt requests

## New Scheduler Lifecycle

The scheduler now has a more explicit lifecycle:

```text
waiting -> prefill -> decoding -> finished
```

Admission now means:

```text
reserve KV
place request in active slot
mark request as prefill
```

Execution now happens from the scheduler step loop:

```text
drain queue
admit waiting requests
run scheduled prefill work
run decode-ready requests
record metrics/history
```

This separation is important because it makes admission and execution distinct scheduler decisions.

## Work Plan Abstraction

The sprint introduced a scheduler work-plan abstraction:

```text
ScheduledRequestWork
```

A work item records:

```text
request_state
num_scheduled_tokens
kind
```

where `kind` is either:

```text
prefill
decode
```

This lets the scheduler describe work before executing it.

Decode work currently schedules one token per request. Prefill work can schedule multiple prompt tokens, and now supports a token budget.

This gives the scheduler a common language for future policies:

```text
which requests should receive work this step?
how many tokens should each receive?
how much prefill vs decode work was scheduled?
how much work was actually executed?
```

## Chunked Prefill Milestone

The key scheduler-level milestone is now proven by test:

```text
max_scheduled_tokens_per_step = 1
prompt_tokens = 2

step 1:
  admit request
  compute 1 prefill token
  request remains in prefill
  decode does not run

step 2:
  compute final prefill token
  request becomes decoding
  decode runs
```

This is the line between budgeted metadata and real scheduler-level chunked prefill.

Before this milestone, the scheduler could report that it scheduled only one prefill token, while the engine still executed full prefill. After this milestone, the engine-facing path receives the scheduled token count through:

```text
prefill_chunk(request_state, num_tokens, kv_block_manager)
```

That means the scheduled token budget controls request progress.

## Why This Matters

Chunked prefill matters because prefill and decode have different serving characteristics.

Prefill is compute-heavy and processes prompt tokens in bulk. Decode is latency-sensitive and usually advances one token at a time per active request.

Without chunked prefill, a long prompt can block decode progress for other requests. With chunked prefill, the scheduler can break long prompts into smaller pieces and interleave them with decode work.

This is important for:

* reducing time-to-first-token for short prompts
* improving responsiveness under mixed workloads
* preventing long prompts from monopolizing scheduler iterations
* making prefill/decode tradeoffs explicit
* supporting future SLO-aware scheduling

## Current Status

Completed:

```text
✅ EngineScheduler cleaned back down to lifecycle ownership
✅ scheduler_work_plan.py extracted
✅ admission no longer runs full prefill
✅ explicit request lifecycle: waiting -> prefill -> decoding -> finished
✅ prefill-active request filtering
✅ decode-ready request filtering
✅ prefill work represented as ScheduledRequestWork
✅ prefill work summaries exposed through scheduler state
✅ max_scheduled_tokens_per_step affects prefill work planning
✅ fake-engine chunked prefill semantics pass tests
✅ decode is blocked until prefill_tokens_remaining == 0
```

The scheduler now supports chunked prefill semantics.

## Next Backend Goal

The next goal is to implement real chunked prefill in the Llama backend.

The current Llama prefill path performs full prompt prefill:

```text
tokenize full prompt
run full prompt forward
write all K/V into the physical KV pool
set next_token from final prompt logits
set num_computed_tokens = prompt_tokens
```

Chunked Llama prefill needs to split this into incremental work:

```text
ensure prompt tokens exist
compute start = num_computed_tokens
compute end = min(start + num_tokens, prompt_tokens)
slice input_ids[start:end]
run model forward for only that chunk
write produced K/V into the KV pool at the correct token offset
advance num_computed_tokens
set next_token only after the final prompt chunk
```

The hardest correctness requirement is preserving absolute token positions and KV-cache continuity across chunks.

## Backend Risks

The real Llama implementation has more risk than the fake scheduler implementation because it must preserve:

* prompt token offsets
* position IDs or cache positions
* attention visibility into previous prompt chunks
* correct K/V writes into the paged KV pool
* correct `next_token` generation only after final prompt token
* compatibility with the existing decode path

The first fake-engine milestone proves scheduler semantics. The Llama milestone proves backend correctness.

## Sprint Completion Criteria

This sprint is complete when:

```text
1. Scheduler-level chunked prefill tests pass
2. Llama prefill code is refactored into reusable tokenization / forward / KV-write helpers
3. CustomLlamaDecodeEngine exposes prefill_chunk(...)
4. A small TinyLlama smoke test proves a prompt can be prefetched across chunks
5. Decode starts only after final prompt chunk
6. Existing runtime tests and benchmark smoke still pass
```

Stretch goal:

```text
Benchmark mixed long-prompt and short-prompt workloads and show that chunked prefill improves scheduler responsiveness / TTFT behavior compared with full-prefill admission.
```

## Recruiting Narrative

This sprint turns the project from a simple continuous batching runtime into a more realistic LLM serving system.

The story is:

```text
I built a single-node LLM inference runtime with paged KV cache management,
continuous batching, custom CUDA paged attention, and token-budgeted chunked
prefill scheduling. I separated admission from execution, modeled prefill and
decode as scheduler-owned token work, and implemented request lifecycle
transitions so long prompts can be incrementally prefetched without blocking
decode-ready requests.
```

That is the core technical narrative for AI infrastructure, inference systems, and kernel-adjacent internship interviews.
