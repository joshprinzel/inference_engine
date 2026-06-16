# DecodeBatch Invariants

## Why This Note Exists

`DecodeBatch` is the boundary between the scheduler/runtime side and the attention backend side.

It is the object that says:

> These are the active requests, these are their current tokens, these are their KV cache locations, and this is what the attention backend is allowed to read.

This note pins down the invariant so we do not keep creating off-by-one bugs between the scheduler and the CUDA kernel.

---

## The Core Invariant

For decode-before-write execution:

```python
position = request_state.prompt_tokens + request_state.generated_tokens
seq_len = position
```

Meaning:

```text
position = where the next token's KV will be written
seq_len  = how many KV tokens already exist and are visible to attention
```

The attention backend should only attend over KV that already exists.

The next token's KV is written **after** the decode/model step.

---

## Example

Suppose a request has:

```text
prompt_tokens = 512
generated_tokens = 0
```

Then:

```text
position = 512
seq_len  = 512
```

The attention backend can read existing KV positions:

```text
0 ... 511
```

After the decode step, the runtime writes the newly generated token's KV at:

```text
position = 512
```

Then the request becomes:

```text
prompt_tokens = 512
generated_tokens = 1
```

Next step:

```text
position = 513
seq_len  = 513
```

Now attention can read:

```text
0 ... 512
```

---

## The Bug We Found

The old logic was:

```python
position = request_state.prompt_tokens + request_state.generated_tokens
seq_len = position + 1
```

That was wrong for decode-before-write execution.

Why?

Because `seq_len = position + 1` means attention is allowed to read the token at `position`.

But the token at `position` has not been written yet.

That is how we got this failure in the Python reference path:

```text
IndexError: token_position=128 maps to logical_block_index=16,
but block_table only has 16 blocks
```

The reference backend caught the bug because it validates every token lookup through `KVCacheLayout.locate_token`.

The CUDA backend did not catch it as cleanly because kernels generally assume valid metadata. That makes this invariant even more important.

---

## DecodeBatch Fields

A `DecodeBatch` contains:

```python
request_ids: list[str]
input_token_ids: torch.Tensor
positions: torch.Tensor
seq_lens: torch.Tensor
block_tables: torch.Tensor
```

The two fields that matter most for this invariant are:

```text
positions
seq_lens
```

### `positions`

`positions[i]` is the token position where request `i` will write the next generated token's KV.

It is a write position.

### `seq_lens`

`seq_lens[i]` is the number of KV tokens request `i` already has available for attention.

It is a read length.

These are related, but they are not “position plus one” in decode-before-write mode.

They are equal:

```python
seq_len = position
```

---

## Relationship to Block Tables

Each request owns a block table:

```text
logical block index -> physical block id
```

For any token position:

```python
logical_block_index = token_position // block_size
block_offset = token_position % block_size
physical_block_id = block_table[logical_block_index]
```

The CUDA kernel consumes padded tensor block tables:

```text
[batch_size, max_blocks_per_seq]
```

The Python reference backend converts those padded tensor block tables back to Python lists and ignores `-1` padding.

---

## Relationship to KV Allocation

Before writing generated-token KV, the runtime must ensure capacity for the write position:

```python
token_position = request_state.prompt_tokens + request_state.generated_tokens

kv_block_manager.ensure_capacity_for_token(
    request_id=request_state.request_id,
    token_position=token_position,
)
```

Then the runtime writes:

```python
cache_pool.write_request_token(
    layer_id=layer_id,
    block_table=block_table,
    token_position=token_position,
    key=key,
    value=value,
)
```

Then it increments:

```python
request_state.generated_tokens += 1
```

The order matters:

```text
1. attention reads existing KV
2. runtime ensures capacity for new KV
3. runtime writes new KV
4. generated_tokens increments
```

---

## Correct Decode Step Timeline

For one request:

```text
Start of step:
    prompt_tokens = P
    generated_tokens = G

Build DecodeBatch:
    position = P + G
    seq_len  = P + G

Attention:
    reads KV positions [0, seq_len - 1]

After attention/model step:
    ensure capacity for token_position = position
    write generated token KV at position
    generated_tokens += 1
```

That is the invariant.

---

## Why This Matters

This invariant protects the boundary between:

```text
scheduler/runtime code
KV block manager
KV cache pool
attention backend
CUDA kernel
```

If `seq_lens` is too small, the model ignores valid context.

If `seq_lens` is too large, the backend reads KV that does not exist.

For the Python reference path, that becomes an index error.

For the CUDA path, that can become silent wrong memory access.

---

## Validated By

These tests now pass with the corrected invariant:

```bash
python experiments/test_decode_batch_attention_backend.py \
  --batch-size 4 \
  --seq-len 128 \
  --num-query-heads 16 \
  --num-kv-heads 4 \
  --head-dim 128
```

```bash
python experiments/test_decode_batch_attention_backend.py \
  --batch-size 32 \
  --seq-len 512 \
  --num-query-heads 16 \
  --num-kv-heads 4 \
  --head-dim 128 \
  --total-blocks 4096 \
  --iters 50
```

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

---

## Interview Explanation

The important invariant in my `DecodeBatch` is that `seq_lens` represents the visible KV length, while `positions` represents the next KV write position.

In decode-before-write execution, those are equal:

```python
position = prompt_tokens + generated_tokens
seq_len = position
```

The attention backend reads only existing KV tokens. After the decode step, the runtime writes the newly generated token's KV at `position`, then increments `generated_tokens`.

That invariant matters because an off-by-one at this boundary can either truncate context or cause the CUDA backend to read KV entries that have not been written.