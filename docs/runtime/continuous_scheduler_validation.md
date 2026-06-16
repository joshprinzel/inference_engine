# Continuous Scheduler Validation

## Goal

Validate that the real inference server supports iteration-level scheduling semantics.

The specific behaviors under test are:

1. Requests can arrive while other requests are already decoding
2. New requests can be admitted into available execution slots
3. Requests with shorter max_new_tokens can finish earlier than longer requests
4. Finished requests free their slots independently.
5. Per-request generated token accounting matches the requested generation length.


## Scheduler Configuration

Scheduler-type: continuous_slots

Execution Strat:

- Fixed number of logical slots
- Each active request owns its own KV state.
- Scheduler drains the external queue each engine step
- Waiting requests are admitted into free slots
- Occupied slots decode one token per engine step
- Finished requests are removed from their slot immediately.

This scheduler prioritizes correctness over batch GPU performance.

___

## Early Finish Experiment

Requests:

| Request | max_new_tokens |
|---|---:|
| 0 | 8 |
| 1 | 16 |
| 2 | 32 |
| 3 | 32 |

Expected behavior:
- The 8-token request finishes first
- The 16-token request finishes second
- The 32-token request finishes third
- All slots are eventually free.


Observed Behavior:
request_id | max_new_tokens | ttft | latency | output_chars
--- | --- | --- | --- | ---
0 | 8 | 1.0017 | 1.2732 | 24
1 | 16 | 1.0203 | 1.6212 | 47
2 | 32 | 1.0451 | 2.0709 | 81
3 | 32 | 1.0671 | 2.0927 | 81

Key Result:
- The shorter requests completed earlier while longer requests continued decoding. 



## Late Arrival Experiment

Requests:

| Request | Arrival delay | max_new_tokens |
|---|---:|---:|
| 0 | 0.00s | 64 |
| 1 | 0.00s | 64 |
| 2 | 0.50s | 16 |
| 3 | 0.75s | 16 |

Expected generated tokens:
64 + 64 + 16 + 16 = 160

Observed `/metrics_json` result:

request_id | delay_seconds | max_new_tokens | ttft | latency | output_chars
--- | --- | --- | --- | --- | ---
0 | 0.00 | 64 | 0.2463 | 2.4344 | 164
1 | 0.00 | 64 | 0.3114 | 2.4536 | 164
2 | 0.50 | 16 | 0.1517 | 0.8088 | 47
3 | 0.75 | 16 | 0.1857 | 0.8092 | 47
___

# What This Proves

The server now implements the core control-plane semantics of an LLM serving engine:

- queueing
- admission
- active set management
- decode iteration scheduling
- independent request completion
- slot freeing
- per-request latency and token accounting

This is distinct from the static microbatch scheduler, which was used to validate batched GPU execution.

## Known Limitation

The continuous scheduler currently uses per-request KV state and per-request decode calls.

That means it validates scheduling semantics, but it does not yet provide the GPU efficiency of a true continuous batching engine.

The next layer solves this by making KV memory a first-class resource.

## Transition to Layer 2

Layer 2 begins by replacing abstract request slots with explicit KV cache allocation.

The next system component is a KV block manager:

- allocate blocks for a request
- append blocks as generation grows
- free blocks when a request finishes
- track block table metadata
- expose KV memory usage and fragmentation metrics
- use KV capacity during admission decisions
