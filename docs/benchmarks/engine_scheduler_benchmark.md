# Engine Scheduler Benchmark

## Summary

This benchmark validates the engine-agnostic scheduler runtime.

The benchmark runs two decode engines under the same `EngineScheduler`:

```text
HFDecodeEngine
    Real text generation through Hugging Face.

SyntheticCudaDecodeEngine
    Synthetic model math, but real DecodeBatch lowering, paged KV cache layout,
    KV block management, AttentionBackend dispatch, and CUDA paged attention execution.
```

The purpose is not to compare Hugging Face full-model generation against the synthetic CUDA backend as equivalent workloads.

The purpose is to prove that the same scheduler can drive multiple decode engines through the same request lifecycle and metrics path.

---

## Runtime Path

```text
RequestQueue
    -> EngineScheduler
    -> DecodeEngine
        -> HFDecodeEngine
        -> SyntheticCudaDecodeEngine
```

For the synthetic CUDA path:

```text
EngineScheduler
    -> SyntheticCudaDecodeEngine
    -> DecodeBatch
    -> KVCachePool
    -> AttentionBackend
    -> CUDA paged attention kernel
```

---

## Benchmark Configuration

### HFDecodeEngine

```text
num_requests: 1
max_slots: 1
prompt_tokens: 7
max_new_tokens: 16
```

### SyntheticCudaDecodeEngine

```text
batch sizes: 1, 4, 8, 16, 32
prompt_tokens: 128, 512
max_new_tokens: 32
query heads: 16
kv heads: 4
head_dim: 128
attention backend: cuda
```

---

## Results

| Engine | Requests | Prompt Tokens | Max New Tokens | Wall Seconds | Tokens Generated | Tokens/s | Decode Steps | Decode Batches | Backend Median ms | Backend Min ms | Backend Max ms | Finished |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hf | 1 | 7 | 16 | 1.1411 | 16 | 14.02 | 16 | 0 | N/A | N/A | N/A | true |
| synthetic-cuda | 1 | 128 | 32 | 0.0377 | 32 | 848.24 | 32 | 32 | 0.0860 | 0.0686 | 0.8428 | true |
| synthetic-cuda | 4 | 128 | 32 | 0.0836 | 128 | 1531.29 | 32 | 32 | 0.0906 | 0.0666 | 0.2171 | true |
| synthetic-cuda | 8 | 128 | 32 | 0.1326 | 256 | 1930.42 | 32 | 32 | 0.0952 | 0.0686 | 0.1597 | true |
| synthetic-cuda | 16 | 128 | 32 | 0.2397 | 512 | 2135.92 | 32 | 32 | 0.1316 | 0.1178 | 0.1782 | true |
| synthetic-cuda | 32 | 128 | 32 | 0.4868 | 1024 | 2103.46 | 32 | 32 | 0.2161 | 0.1761 | 0.2877 | true |
| synthetic-cuda | 1 | 512 | 32 | 0.0378 | 32 | 847.67 | 32 | 32 | 0.0881 | 0.0625 | 0.4137 | true |
| synthetic-cuda | 4 | 512 | 32 | 0.0811 | 128 | 1578.39 | 32 | 32 | 0.0906 | 0.0676 | 0.1884 | true |
| synthetic-cuda | 8 | 512 | 32 | 0.1374 | 256 | 1863.05 | 32 | 32 | 0.1029 | 0.0799 | 0.1843 | true |
| synthetic-cuda | 16 | 512 | 32 | 0.2388 | 512 | 2144.00 | 32 | 32 | 0.1418 | 0.1096 | 0.3215 | true |
| synthetic-cuda | 32 | 512 | 32 | 0.5056 | 1024 | 2025.44 | 32 | 32 | 0.2263 | 0.1853 | 0.2826 | true |

---

## Observations

### 1. The HF path works as the real text-generation baseline

The HF case generated 16 tokens in 1.141 seconds, or about 14.02 tokens/s.

This path is valuable because it proves the server and scheduler can still drive real model output.

However, the HF path does not use the custom CUDA paged attention backend. Hugging Face owns the actual model internals and `past_key_values`.

---

### 2. The synthetic CUDA path successfully exercises the custom runtime path

Every synthetic CUDA benchmark case completed successfully.

For all synthetic CUDA cases:

```text
all_finished: true
decode_iterations: 32
decode_batches_built: 32
kv_used_blocks after finish: 0
```

This means each decode step successfully built a `DecodeBatch`, executed the backend, emitted synthetic tokens, and freed KV blocks after request completion.

That is the main correctness signal.

---

### 3. Throughput scales with batch size until scheduler/runtime overhead dominates

For prompt length 128:

```text
batch=1:   848 tok/s
batch=4:  1531 tok/s
batch=8:  1930 tok/s
batch=16: 2136 tok/s
batch=32: 2103 tok/s
```

For prompt length 512:

```text
batch=1:   848 tok/s
batch=4:  1578 tok/s
batch=8:  1863 tok/s
batch=16: 2144 tok/s
batch=32: 2025 tok/s
```

The synthetic CUDA path improves as batching increases from 1 to 16 requests. At 32 requests, throughput flattens or slightly drops.

That is expected for this current implementation because the benchmark includes Python scheduler overhead, synthetic tensor generation, KV writes, DecodeBatch construction, and CUDA backend execution.

---

### 4. Backend kernel time remains small relative to end-to-end wall time

Backend median times are sub-millisecond across the synthetic cases:

```text
batch=1, prompt=128:   0.0860 ms
batch=16, prompt=128:  0.1316 ms
batch=32, prompt=128:  0.2161 ms

batch=1, prompt=512:   0.0881 ms
batch=16, prompt=512:  0.1418 ms
batch=32, prompt=512:  0.2263 ms
```

This suggests that much of the benchmark wall time is not the CUDA attention kernel itself. The current bottleneck is likely Python-side runtime overhead:

```text
scheduler stepping
DecodeBatch construction
synthetic q generation
synthetic K/V writeback
per-request Python loops
```

This is useful because it tells us where future optimization should go.

---

## What This Benchmark Proves

This benchmark proves:

```text
1. EngineScheduler can drive multiple DecodeEngine implementations.
2. HFDecodeEngine works as a real text-generation baseline.
3. SyntheticCudaDecodeEngine works under the same scheduler.
4. DecodeBatch lowering is exercised on every synthetic CUDA decode step.
5. KVBlockManager allocates and frees blocks correctly.
6. KVCachePool and AttentionBackend integrate with the scheduler path.
7. The custom CUDA paged attention backend runs inside the runtime path.
```

The important project milestone is:

> The custom CUDA backend is no longer only an isolated kernel benchmark. It is now exercised through the same scheduler/runtime path as the Hugging Face engine.

---

## What This Benchmark Does Not Prove

This benchmark does not prove:

```text
full custom model execution
real Q/K/V projection through the CUDA path
real logits through the CUDA path
real sampling through the CUDA path
replacement of Hugging Face internals
production scheduler performance
```

The synthetic CUDA engine still uses fake Q tensors, fake generated token IDs, and synthetic K/V writes.

That is intentional at this stage.

---

## Current Project Claim

The current accurate project claim is:

> Built an engine-agnostic LLM inference serving runtime where the same continuous scheduler can drive either a Hugging Face decode engine for real text generation or a synthetic CUDA decode engine that exercises DecodeBatch lowering, paged KV cache management, and a custom CUDA paged attention backend.

---

## Next Steps

The next engineering steps are:

```text
1. Add backend timing history directly to EngineScheduler.
2. Add a benchmark report generator so this markdown can be regenerated from CSV.
3. Add a server-level benchmark for /generate_stream.
4. Begin planning CustomModelDecodeEngine.
```

The future `CustomModelDecodeEngine` should replace the synthetic CUDA engine's fake Q/K/V path with real model execution:

```text
token embedding
Q/K/V projection
KVCachePool writeback
CUDA paged attention
MLP
logits
sampling
streamed token output
```