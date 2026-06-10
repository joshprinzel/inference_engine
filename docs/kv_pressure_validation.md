# KV Pressure Validation

## Goal

Validate that the continuous scheduler treats KV cache memory as a finite paged resource.

This experiment checks whether the server can:

- allocate KV blocks for active requests
- grow request block tables during decode
- detect decode-time KV pressure
- avoid deadlock when all active requests need more KV memory
- evict a request to free blocks
- allow remaining requests to continue
- return the KV allocator to a clean state

## Setup

Scheduler:

continuous_slots

KV block configuration:

total_blocks = 16
block_size_tokens = 16

Total token capacity:

16 blocks * 16 tokens/block = 256 tokens

Workload:

4 concurrent streaming requests

Each request used:

max_new_tokens = 96

The test script polls /metrics_json while the requests are running and prints final request status after completion.

## Policy Under Test

The scheduler uses dynamic paged KV growth.

At admission:

A request only needs enough KV blocks for its prompt.

During decode:

Before decoding the next token, the scheduler computes the token position:

token_position = prompt_tokens + generated_tokens

The KV block manager checks whether the request already has a block for that token position.

If the token position crosses a block boundary and a free block exists, the manager appends a physical block to the request's block table.

If no free block exists, the request stalls for that decode step.

If every active request stalls and no progress is possible, the scheduler evicts one active request to break the memory deadlock.

On finish or eviction:

The request's KV blocks are freed and returned to the global free list.

## Observed KV Pressure

During the run, KV usage increased until the cache was full.

Observed progression:

used_blocks rose to full capacity:

6 -> 8 -> 12 -> 16

free_blocks fell to zero:

10 -> 8 -> 4 -> 0

At this point, all physical KV blocks were allocated.

Then active request count dropped as the scheduler evicted or completed requests:

active 4 -> 3 -> 2 -> 1 -> 0

Blocks were returned to the free list:

used_blocks 16 -> 15 -> 13 -> 7 -> 0

Final allocator state:

used_blocks = 0
free_blocks = 16
active_requests = 0

This confirms that KV blocks are allocated during decode and freed when requests leave the system.

## Final Engine Metrics

Final pressure counters:

decode_iterations = 58
decode_stalls = 23
kv_allocation_failures = 23
kv_oom_evictions = 2

Interpretation:

The scheduler encountered 23 decode steps where a request needed another KV block but no free block was available.

The system performed 2 OOM evictions to break decode-time memory deadlock.

After evictions, the remaining requests continued and the KV allocator returned to a clean state.

## Request Outcomes

Final request outcomes:

| Request | max_new_tokens | generated_tokens | status | error |
|---|---:|---:|---|---|
| 0 | 96 | 44 | failed | KV cache exhausted: request evicted to break decode-time memory deadlock |
| 1 | 96 | 60 | failed | KV cache exhausted: request evicted to break decode-time memory deadlock |
| 2 | 96 | 96 | finished | None |
| 3 | 96 | 96 | finished | None |

Two requests were evicted under KV pressure.

Two requests completed successfully after memory was freed.

## What This Proves

The server now models KV memory as a finite paged resource.

The important behaviors are:

- requests own block tables
- physical KV blocks are allocated on demand
- block usage grows during decode
- KV exhaustion can stall decode
- the scheduler detects no-progress deadlock
- OOM eviction frees KV blocks
- remaining requests can continue
- all blocks return to the free pool after completion

This is the first point where the server's scheduling behavior is constrained by KV memory capacity instead of only by request slots.

## Known Limitations

The current implementation is metadata-only.

HF past_key_values still store the real model KV cache. The KVBlockManager tracks the future paged-KV layout, but the attention backend does not yet read from the block tables.

The OOM policy is intentionally simple.

When all active requests are stalled, the scheduler evicts one request. A production engine would use a more sophisticated policy such as preemption, recomputation, CPU swap, priority-aware eviction, or chunked prefill/decode budgeting.

This experiment is not a latency benchmark.

The current backend decodes requests individually, so TTFT and total latency reflect Python/HF execution overhead as well as memory pressure behavior.

## Why This Matters

This experiment validates the Layer 2 control-plane contract for paged KV memory.

The scheduler now has enough information to reason about:

- how many KV blocks are free
- which request owns which physical blocks
- when a request needs a new block
- when memory pressure prevents decode progress
- which blocks can be freed when a request exits

This prepares the engine for a future CUDA C++ paged attention backend.

## Next Step

The next step is DecodeBatch lowering.

The scheduler should convert active requests into kernel-shaped metadata:

- input_token_ids
- positions
- seq_lens
- block_tables

This will define the ABI between the Python scheduler and the future CUDA C++ backend.