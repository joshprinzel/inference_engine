# LLM Inference Systems Lab

A single-node LLM inference serving runtime for studying the systems behind modern LLM serving engines: paged KV cache management, continuous batching, scheduler-owned prefill/decode execution, CUDA-backed paged attention, and Sarathi-style chunked prefill.

This project is not a chatbot wrapper or API application. It is a systems project focused on how an inference server manages GPU memory, schedules token work, and keeps decode progress responsive while handling long prompt prefill.

## Overview

Modern LLM serving systems are constrained by GPU memory, KV cache growth, request scheduling, and the mismatch between prompt prefill and token-by-token decode.

This project implements a miniature inference serving stack around those constraints. The runtime supports request admission, paged KV cache allocation, prefill execution, decode batching, physical KV cache storage, and CUDA-backed paged attention.

The main completed feature is **token-budgeted chunked prefill**. Instead of letting a long prompt monopolize the scheduler during full-prompt prefill, the scheduler splits prefill into bounded chunks. This allows decode work to continue making progress between long-prompt prefill chunks.

## System Architecture

```text
Request Queue
    -> EngineScheduler
        -> admission
        -> active request slots
        -> KV block reservation
        -> prefill work planning
        -> decode-ready filtering
        -> batched decode execution
        -> request cleanup
    -> DecodeEngine
        -> TinyLlama prefill
        -> chunked prompt execution
        -> paged KV cache writes
        -> CUDA-backed paged attention decode
```

The runtime is organized around a clean separation between scheduling policy, request state, KV memory management, and model execution.

## Core Components

### EngineScheduler

`EngineScheduler` owns the serving lifecycle.

It is responsible for:

* draining the external request queue
* admitting waiting requests into active slots
* reserving KV cache blocks
* tracking request lifecycle state
* scheduling prefill work
* filtering decode-ready requests
* executing decode batches
* recording runtime snapshots and scheduler metrics
* cleaning up finished or failed requests

The scheduler is engine-agnostic. It does not own model internals, token sampling, or CUDA-specific implementation details.

### Request Lifecycle

Requests move through an explicit lifecycle:

```text
waiting -> prefill -> decoding -> finished
```

Earlier versions of the runtime coupled admission and execution. A request would be admitted and immediately run full prompt prefill.

The current runtime separates those concerns:

```text
admission reserves resources
prefill materializes prompt KV
decode generates output tokens
cleanup frees KV blocks
```

This separation makes prefill a scheduler-owned unit of work rather than an admission side effect.

### Scheduler Work Plan

The scheduler represents token work using a work-plan abstraction:

```text
ScheduledRequestWork(
    request_state,
    num_scheduled_tokens,
    kind,
)
```

where `kind` is either:

```text
prefill
decode
```

This lets the scheduler reason about prefill and decode as explicit token-level work.

Decode currently schedules one token per request. Prefill can schedule multiple prompt tokens, bounded by a configurable token budget.

This abstraction supports:

* token-budgeted prefill
* decode-ready filtering
* work-plan summaries
* scheduler observability
* future prefill/decode interleaving policies

### KVBlockManager

`KVBlockManager` owns logical KV cache allocation.

It supports:

* checking whether a request can reserve enough KV capacity
* allocating block tables for requests
* ensuring token capacity during decode
* freeing request-owned KV blocks
* reporting KV utilization

This models the resource-management layer needed by paged KV cache serving systems.

### KVCachePool

`KVCachePool` is the physical storage backing paged KV cache.

Prompt and decode K/V tensors are written into this pool according to each request’s block table. Decode then reads from the paged layout during custom attention execution.

### Custom Llama Decode Engine

The custom Llama backend runs TinyLlama-style inference while integrating with the runtime’s scheduler and KV cache system.

It supports:

* prompt tokenization
* full prefill compatibility
* chunked prefill
* request-local prefill KV continuity
* incremental writes into the physical paged KV pool
* paged-attention decode
* next-token setup after prefill completion

### CUDA Paged Attention

The decode path supports a CUDA-backed paged attention backend.

The backend is used during decode to attend over request KV stored in the physical paged KV cache pool. This provides a concrete execution path for studying how block tables, sequence lengths, token positions, and physical KV layout interact during batched decode.

## Chunked Prefill

The main completed feature is **Sarathi-style token-budgeted chunked prefill**.

Full-prompt prefill can produce large scheduler stalls because prompt prefill is much heavier than one-token decode. Chunked prefill splits long prompt computation into smaller scheduler-visible units.

Example with a 20-token prompt and a 4-token prefill budget:

```text
step 1: compute prompt tokens 0-3
step 2: compute prompt tokens 4-7
step 3: compute prompt tokens 8-11
step 4: compute prompt tokens 12-15
step 5: finish prefill, transition to decode
```

The scheduler tracks:

```text
prompt_tokens
num_computed_tokens
prefill_tokens_remaining
decode_tokens_remaining
estimated_total_tokens_remaining
```

Decode is guarded so prefill-incomplete requests cannot enter decode.

A request becomes decode-ready only when:

```text
status == "decoding"
prefill_tokens_remaining == 0
```

## Real Backend Chunked Prefill

Chunked prefill is implemented in the real Llama backend, not only in scheduler metadata.

For each scheduled prefill chunk, the custom Llama engine:

1. tokenizes the prompt once
2. computes the chunk range from `num_computed_tokens`
3. runs the model forward only on the scheduled prompt slice
4. preserves prompt continuity using request-local `past_key_values`
5. extracts only the newly computed K/V suffix
6. writes that suffix into the physical paged KV cache pool
7. advances `num_computed_tokens`
8. sets `next_token` only after the final prompt chunk
9. drops duplicate contiguous prefill cache after prompt completion

This allows the scheduler’s prefill token budget to drive real model execution and physical KV materialization.

## End-to-End Smoke Test

The runtime includes an end-to-end smoke test for:

```text
request queue
  -> scheduler admission
  -> chunked Llama prefill
  -> incremental paged KV writes
  -> transition to decoding
  -> CUDA-backed paged attention decode
  -> request completion
```

Observed behavior with a 20-token prompt and 4-token prefill budget:

```text
step 1: status=prefill, num_computed_tokens=4,  prefill_remaining=16
step 2: status=prefill, num_computed_tokens=8,  prefill_remaining=12
step 3: status=prefill, num_computed_tokens=12, prefill_remaining=8
step 4: status=prefill, num_computed_tokens=16, prefill_remaining=4
step 5: decode work executes, num_computed_tokens=21, generated_tokens=1
```

Final state:

```text
status=finished
prompt_tokens=20
generated_tokens=1
num_computed_tokens=21
```

The consistency check is:

```text
num_computed_tokens == prompt_tokens + generated_tokens
```

## Sarathi-Style Chunked Prefill Benchmark

The benchmark evaluates the prefill/decode interference problem that motivates Sarathi-style chunked prefill.

Workload:

1. Start one short request.
2. Let the short request enter decode.
3. Admit one long-prompt request.
4. Measure time-between-tokens for the already-decoding short request.

This measures whether long prompt prefill stalls ongoing decode work.

### Benchmark Result

| Mode            | Prefill Budget | Long Prompt Tokens | Avg Short TBT | P50 Short TBT | P95 Short TBT | Worst Short TBT | Total Runtime |
| --------------- | -------------: | -----------------: | ------------: | ------------: | ------------: | --------------: | ------------: |
| Full prefill    |      unlimited |              1,942 |      115.4 ms |       20.3 ms |       27.8 ms |      4,480.8 ms |    5,940.8 ms |
| Chunked prefill |      16 tokens |              1,942 |       89.2 ms |       86.5 ms |      124.6 ms |        136.5 ms |    8,729.0 ms |

Chunked prefill reduced worst-case short-request time-between-tokens from **4.48s to 136.5ms**, a **97.0% reduction**, by breaking long prompt prefill into 16-token chunks.

The tradeoff is visible: total benchmark runtime increases because chunked prefill performs repeated smaller forward passes. The benefit is that ongoing decode work no longer experiences one catastrophic multi-second stall.

## Testing

The project uses layered tests and smoke scripts.

Coverage includes:

* scheduler admission lifecycle
* request status transitions
* prefill-active filtering
* decode-ready filtering
* scheduler work-plan construction
* token-budgeted prefill planning
* fake-engine chunked prefill behavior
* real Llama first-chunk prefill
* real Llama multi-chunk prefill
* next-token setup after final prefill chunk
* physical paged KV writes during prefill
* end-to-end scheduler + chunked prefill + paged decode smoke
* Sarathi-style TBT benchmark for prefill/decode interference

The testing strategy separates:

```text
scheduler correctness
backend correctness
end-to-end runtime behavior
benchmark evidence
```

## Current Status

Completed:

```text
✅ single-node inference runtime skeleton
✅ request queue and active slots
✅ paged KV block manager
✅ physical KV cache pool
✅ custom TinyLlama decode engine
✅ CUDA-backed paged attention decode path
✅ continuous batching scheduler
✅ explicit request lifecycle: waiting -> prefill -> decoding -> finished
✅ admission separated from execution
✅ scheduler work-plan abstraction
✅ token-budgeted prefill scheduling
✅ real Llama backend chunked prefill
✅ incremental K/V writes into physical paged KV cache
✅ end-to-end chunked prefill + paged decode smoke
✅ Sarathi-style TBT benchmark with measured stall reduction
```

Future work:

```text
- decode-maximal batching policy
- SLO-aware scheduling
- short-prefill prioritization
- full prefill vs chunked prefill correctness comparison across more prompts
- deeper benchmark sweeps across budgets and prompt lengths
- dashboard polish
- distributed control-plane extension
```

## Technical Stack

```text
Python
PyTorch
Hugging Face Transformers
CUDA
C++
pytest
Nsight / compute-sanitizer workflow
```

## Why This Project Matters

LLM inference performance is not only a model problem. Serving systems must manage KV memory, schedule heterogeneous token work, and preserve decode responsiveness under long-prompt pressure.

This project implements those concerns directly:

* paged KV cache allocation
* physical KV cache storage
* active request lifecycle management
* continuous batching
* scheduler-owned prefill/decode execution
* chunked prompt prefill
* CUDA-backed paged attention decode
* benchmarked prefill/decode interference

The result is a compact but realistic inference systems lab for studying the mechanisms behind production LLM serving engines.



## Streamlit Runtime Playground

The project includes a Streamlit playground for inspecting runtime behavior interactively.

The playground is useful for visualizing:

* request admission
* active scheduler slots
* waiting / prefill / decoding / finished request states
* KV cache utilization
* decode batch behavior
* scheduler policy behavior
* chunked prefill progress
* work-plan summaries
* generated text output

Run from the repository root:

```bash
streamlit run experiments/dashboard/app.py
```

If imports fail, run with the project root on `PYTHONPATH`:

```bash
PYTHONPATH=. streamlit run experiments/dashboard/app.py
```

The dashboard supports experimenting with runtime configuration such as:

```text
number of requests
prompt length
max new tokens
scheduler policy
maximum active slots
decode batch budget
prefill token budget
attention backend
```

For chunked prefill demos, set a small prefill token budget, such as:

```text
prefill token budget = 4, 8, or 16
```

Then inspect how requests move through:

```text
waiting -> prefill -> decoding -> finished
```

and how prefill work is split across scheduler steps before decode begins.

