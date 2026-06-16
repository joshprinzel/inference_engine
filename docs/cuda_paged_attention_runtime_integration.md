# Development Sprint Milestones: CUDA Paged Attention Runtime Integration

## Sprint Goal

Move the project from an isolated CUDA paged attention kernel into a runtime-shaped inference system component.

The goal of this sprint is not full Hugging Face model replacement yet. The goal is to prove that the custom CUDA paged attention backend can be driven by the same runtime metadata a real inference server would use:

```text
RequestState
  -> KVBlockManager
  -> DecodeBatch
  -> KVCachePool
  -> AttentionBackend
  -> CUDA paged attention kernel
```

This sprint prioritizes recruiting signal: correctness, integration, benchmarking, documentation, and an honest engineering narrative.

---

## Milestone 1: Freeze CUDA Paged Attention Kernel Baseline

### Status

Complete.

### Summary

The CUDA paged attention backend now has a stable baseline implementation, frozen as `v9a`.

The kernel supports:

* Batched decode attention
* Paged KV cache layout
* Ragged sequence lengths
* MHA / GQA / MQA
* One CTA per `(sequence, query_head)`
* Warp-level QK score generation
* Shared-memory score materialization
* Parallel softmax denominator reduction
* In-place softmax probability caching in `scores[]`

### Why this matters

This establishes the CUDA backend as a usable runtime component instead of an endlessly changing kernel experiment.

The project now has a defensible kernel evolution story:

```text
v1: correctness-first single request kernel
v2: threaded per-head kernel
v3: shared score reuse
v4: batched decode
v5: MHA/GQA/MQA support
v6a: parallel softmax denominator
v7: warp-level QK score generation
v8: online softmax experiment, correct but slower
v9a: frozen baseline with probability caching
```

### Decision

`v9a` is the current default CUDA paged attention backend.

Further CUDA-native optimization is deferred until after runtime integration.

---

## Milestone 2: Add Attention Backend Abstraction

### Status

Complete.

### Summary

Added an `AttentionBackend` abstraction with two implementations:

```text
ReferencePagedAttentionBackend
CudaPagedAttentionBackend
```

The reference backend calls the Python paged attention reference.

The CUDA backend calls the compiled `paged_attention_cuda.paged_attention_decode_batch` extension.

### Interface

```python
backend.decode(
    q=q,
    cache_pool=cache_pool,
    layer_id=layer_id,
    block_tables=decode_batch.block_tables,
    seq_lens=decode_batch.seq_lens,
)
```

### Important design decision

The backend interface takes:

```text
KVCachePool
DecodeBatch.block_tables
DecodeBatch.seq_lens
```

instead of raw detached tensors.

This keeps `KVCachePool` as the source of truth for the physical KV cache layout.

### Why this matters

This creates the seam needed to switch between:

```text
reference backend
cuda backend
```

without rewriting scheduler/runtime code.

---

## Milestone 3: DecodeBatch to CUDA Attention Backend Integration Test

### Status

Complete.

### Summary

Added a focused integration test proving that `DecodeBatch` metadata can drive the CUDA backend using the real `KVCachePool` physical layout.

### Validated path

```text
RequestState metadata
  -> KVBlockManager block tables
  -> DecodeBatch tensors
  -> KVCachePool physical key/value tensors
  -> AttentionBackend
  -> CUDA paged attention
```

### Canonical large test

Configuration:

```text
mode: GQA
batch_size: 32
seq_len: 512
num_query_heads: 16
num_kv_heads: 4
head_dim: 128
block_size: 8
```

Result:

```text
max_abs_diff:  0.00012207
mean_abs_diff: 0.00000000
ref_med_ms:    126.416893
cuda_med_ms:   0.640000
speedup:       197.53x
```

### Interpretation

This is not full model speedup.

This is attention backend speedup against the Python reference on the runtime-shaped `DecodeBatch + KVCachePool` integration benchmark.

The honest claim:

> The CUDA paged attention backend was approximately 197x faster than the Python reference on a GQA batch=32, seq_len=512 DecodeBatch integration benchmark.

### Why this matters

This is the first proof that the CUDA kernel is no longer isolated. It is callable through runtime-shaped abstractions.

---

## Milestone 4: Scheduler-Owned Attention Backend Harness

### Status

Complete.

### Summary

Added a repeated decode harness that simulates the scheduler-owned decode loop.

The harness repeatedly:

1. Tracks active `RequestState` objects.
2. Builds a `DecodeBatch`.
3. Creates synthetic query tensors.
4. Calls the selected attention backend.
5. Appends synthetic generated-token KV entries.
6. Grows block tables when needed.
7. Marks finished requests.
8. Frees KV blocks.
9. Emits per-step metrics to CSV.

### Validated path

```text
RequestState
  -> active request filtering
  -> DecodeBatch construction
  -> KVBlockManager block table lookup
  -> KVCachePool physical KV read
  -> CUDA attention backend
  -> generated-token KV append
  -> KV block growth
  -> request completion
  -> KV block freeing
```

### CUDA run

Command:

```bash
python experiments/run_scheduler_attention_backend_harness.py \
  --backend cuda \
  --batch-size 32 \
  --prompt-tokens 512 \
  --max-new-tokens 32 \
  --num-query-heads 16 \
  --num-kv-heads 4 \
  --head-dim 128
```

Observed result:

```text
backend:              cuda
batch_size:           32
prompt_tokens:        512
max_new_tokens:       32
decode_steps:         32
total_tokens_emitted: 1024
wall_seconds:         0.097097
tokens_per_second:    10546.13
backend_med_ms:       0.032768
backend_min_ms:       0.022528
backend_max_ms:       1.036288
final used_blocks:    0
```

### Important limitation

This harness uses:

* Synthetic Q tensors
* Synthetic generated token IDs
* Synthetic generated-token K/V writes

This is intentional.

The harness validates runtime and memory-system plumbing, not full model math.

### Why this matters

This moves the project from one-shot kernel tests to repeated scheduler-style decode execution.

This is a significant systems milestone.

---

## Milestone 5: DecodeBatch Semantic Invariant Fix

### Status

In progress / next.

### Problem

The existing `build_decode_batch` logic used:

```python
position = request_state.prompt_tokens + request_state.generated_tokens
seq_len = position + 1
```

This caused the decode batch to expose one more KV token than had actually been written.

The reference backend caught this correctly with:

```text
IndexError: token_position maps past the allocated block table
```

### Correct invariant

```text
seq_lens = number of KV tokens already visible to attention
positions = token position where the next generated token KV will be written
```

For decode-before-write execution:

```python
position = request_state.prompt_tokens + request_state.generated_tokens
seq_len = position
```

### Why this matters

This is a core runtime invariant.

Decode attention should attend over existing KV entries. The newly generated token's KV is written after the attention/model step.

### Next action

Patch `decode_batch.py` so `build_decode_batch` uses:

```python
position = request_state.prompt_tokens + request_state.generated_tokens
seq_len = position
```

Then update tests that were compensating for the previous `+1` behavior.

---

## Current Project State

The project now has the following working layers:

```text
Layer 1: Hugging Face baseline ModelRunner
Layer 2: Paged KV cache layout and block manager
Layer 3: DecodeBatch lowering
Layer 4: CUDA paged attention backend
Layer 5: AttentionBackend abstraction
Layer 6: DecodeBatch -> CUDA backend integration test
Layer 7: Scheduler-owned synthetic decode harness
```

The Hugging Face model runner is still a black-box baseline and does not yet use the custom CUDA attention kernel.

That is acceptable for this sprint.

The current custom backend is integrated into the runtime-shaped paged attention path, not yet into full model execution.

---

## What This Sprint Proves

This sprint proves:

1. The CUDA paged attention kernel is correct against the Python reference.
2. The kernel supports the project’s physical KV cache layout.
3. Runtime metadata can be lowered into CUDA-compatible tensors.
4. The backend can be selected through an abstraction.
5. The scheduler-style loop can repeatedly call the CUDA backend.
6. KV blocks can grow as tokens are generated.
7. KV blocks are freed when requests finish.
8. Metrics can be emitted per decode step.

This is the first coherent end-to-end runtime story underneath the server.

---

## What This Sprint Does Not Prove Yet

This sprint does not yet prove:

* Full Hugging Face model replacement
* Real Q/K/V projection integration
* Real logits generation through custom layers
* Production-grade scheduler policy
* Chunked prefill
* Prefix caching
* CUDA Graphs
* Native CUDA benchmark harness
* Fully optimized CUDA memory path

Those are future milestones.

---

## Next Sprint Priorities

### Priority 1: Fix DecodeBatch invariant globally

Make `DecodeBatch` semantics correct and documented.

Target invariant:

```text
seq_lens = visible KV length
positions = next KV write position
```

### Priority 2: Add runtime documentation

Create:

```text
docs/runtime/scheduler_attention_backend_harness.md
docs/runtime/decode_batch_invariants.md
```

### Priority 3: Add benchmark report generation

Turn harness CSV into a small report:

```text
backend median latency
tokens/sec proxy
active batch size over time
KV utilization over time
used/free blocks
```

### Priority 4: Move toward scheduler integration

Connect the harness behavior to the actual scheduler path.

Target:

```text
scheduler builds DecodeBatch
scheduler calls AttentionBackend
scheduler records runtime events
```

### Priority 5: Defer deep CUDA optimization

Do not chase `half2`, `cp.async`, Tensor Cores, CUDA Graphs, or compiler integration until the runtime path is documented and benchmarked.

---

## Recruiting Narrative

The concise narrative:

> I built a from-scratch LLM inference systems lab focused on the decode path. The project includes a paged KV cache manager, DecodeBatch lowering, a custom CUDA paged attention backend supporting MHA/GQA/MQA, and a scheduler-owned synthetic decode harness. I used Nsight-driven iteration to optimize the kernel, then integrated it into the runtime-shaped KV/cache/scheduler path and validated correctness against a Python reference.

The stronger technical narrative:

> The key design decision was treating `KVCachePool` and `DecodeBatch` as the boundary between runtime scheduling and GPU execution. The Python reference and CUDA backend share the same interface, so correctness can be checked at the runtime boundary. This let me separate model execution from scheduler/KV/backend integration and prove the paged attention path independently before attempting full model replacement.

---

## Commit Checklist

Recommended commits for this sprint:

```bash
git add attention_backend.py
git commit -m "Add attention backend abstraction"

git add experiments/test_decode_batch_attention_backend.py
git commit -m "Add DecodeBatch attention backend integration test"

git add experiments/run_scheduler_attention_backend_harness.py
git commit -m "Add scheduler attention backend harness"

git add docs/runtime/development_sprint_milestones.md
git commit -m "Document runtime integration sprint milestones"
```

After fixing `decode_batch.py`:

```bash
git add decode_batch.py experiments/test_decode_batch_attention_backend.py
git commit -m "Fix DecodeBatch visible KV length invariant"
```
