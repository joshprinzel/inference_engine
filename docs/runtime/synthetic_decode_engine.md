# Synthetic Decode Engine

## Purpose

`SyntheticDecodeEngine` is a scheduler-owned decode runtime used to validate the paged attention execution path without depending on full Hugging Face model internals.

It is not full model execution.

It uses synthetic:

- query tensors
- generated token IDs
- generated-token K/V writes

The purpose is to validate runtime plumbing:

```text
RequestState
  -> KVBlockManager
  -> DecodeBatch
  -> KVCachePool
  -> AttentionBackend
  -> CUDA paged attention kernel
```

---

## Why This Exists

The current `ModelRunner` uses Hugging Face as a black box.

That path does not expose:

- per-layer Q tensors
- custom paged KV layout
- custom block tables
- direct attention backend selection

So `SyntheticDecodeEngine` exists as the intermediate runtime layer between isolated kernel tests and full model integration.

It proves that the scheduler/KV/backend boundary works before attempting model replacement.

---

## Owned Components

`SyntheticDecodeEngine` owns:

```text
SyntheticDecodeConfig
KVCacheLayout
KVCachePool
KVBlockManager
AttentionBackend
RequestState list
decode loop
per-step metrics
```

The engine initializes prompt KV, repeatedly builds `DecodeBatch`, calls the selected backend, writes generated-token KV, and frees blocks when requests finish.

---

## Decode Step

Each decode step does:

```text
1. Select active requests.
2. Build DecodeBatch.
3. Create synthetic q tensor.
4. Call AttentionBackend.decode(...).
5. For each active request:
      compute next KV write position
      ensure KV block capacity
      write synthetic generated-token KV
      increment generated_tokens
      update fake next_token
      finish request if max_new_tokens reached
6. Record per-step metrics.
```

---

## DecodeBatch Invariant

The engine relies on the project-level `DecodeBatch` invariant:

```text
positions = next KV write position
seq_lens  = visible KV length
```

For decode-before-write:

```python
position = request_state.prompt_tokens + request_state.generated_tokens
seq_len = position
```

The attention backend reads existing KV first.

The generated token's KV is written after the backend call.

---

## Backend Selection

The engine supports:

```text
reference
cuda
```

The reference backend calls the Python paged attention implementation.

The CUDA backend calls the compiled CUDA paged attention extension.

Both backends consume the same runtime boundary:

```python
backend.decode(
    q=q,
    cache_pool=cache_pool,
    layer_id=layer_id,
    block_tables=decode_batch.block_tables,
    seq_lens=decode_batch.seq_lens,
)
```

---

## Metrics

Each decode step records:

```text
decode_step
active_batch_size
backend
backend_ms
kv_used_blocks
kv_free_blocks
kv_utilization
total_tokens_emitted
```

These metrics are written to:

```text
results/scheduler_attention_backend_harness.csv
```

The report generator turns the CSV into:

```text
docs/runtime/scheduler_attention_backend_harness_report.md
```

---

## CUDA Example

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

This runs a synthetic repeated decode workload using the CUDA paged attention backend.

---

## Reference Example

```bash
python experiments/run_scheduler_attention_backend_harness.py \
  --backend reference \
  --batch-size 4 \
  --prompt-tokens 128 \
  --max-new-tokens 8 \
  --num-query-heads 16 \
  --num-kv-heads 4 \
  --head-dim 128
```

The reference path is slower but useful for validating semantics.

---

## What This Proves

`SyntheticDecodeEngine` proves:

- request states can survive across decode steps
- `DecodeBatch` can be rebuilt every step
- block tables come from `KVBlockManager`
- KV cache data lives in `KVCachePool`
- generated tokens append new KV entries
- block tables grow as sequences cross block boundaries
- finished requests free their KV blocks
- attention backend calls happen inside a repeated scheduler-owned decode loop

---

## What This Does Not Prove

This does not prove:

- full model execution
- real Q/K/V projections
- real logits
- real token sampling
- Hugging Face model replacement
- production scheduler policy

Those are later milestones.

---

## Recruiting Narrative

The concise explanation:

> I built a synthetic decode engine to validate the runtime boundary between scheduling, paged KV cache management, DecodeBatch lowering, and a custom CUDA paged attention backend. This let me test the inference runtime path independently of Hugging Face internals before attempting full model integration.

The important distinction:

> The engine is synthetic at the model-math level, but real at the scheduler/KV/backend boundary.