# Custom Llama Backend v2: KVCachePool-backed Decode

## Goal

This backend version moves the custom TinyLlama decode path from a request-local contiguous K/V cache to the runtime’s paged `KVCachePool`.

The main goal is to make the runtime KV layout the source of truth for real TinyLlama K/V tensors.

This is the bridge between:

```text
v1: contiguous PyTorch past_key_values
```

and:

```text
v3: CUDA paged attention reading KVCachePool + block tables directly
```

The v2 backend still gathers K/V from `KVCachePool` back into contiguous PyTorch tensors before running attention. CUDA paged attention is not wired into the real model path yet.

## Version Ladder

```text
v0: full recompute custom Llama backend                  ✅
v1: contiguous PyTorch KV-cached decode                  ✅
v2a: KVCachePool writeback/gather roundtrip              ✅
v2b: decode logits using KVCachePool-gathered cache      ✅
v2c: CustomLlamaDecodeEngine uses KVCachePool source     ✅
v3: CUDA paged attention in real decode path             next
```

## Runtime Lifecycle

The scheduler remains engine-agnostic.

The request lifecycle is:

```text
RequestQueue
 -> EngineScheduler
 -> KVBlockManager allocation
 -> CustomLlamaDecodeEngine.init_request_state
 -> prompt prefill
 -> K/V writeback into KVCachePool
 -> decode loop
 -> gather K/V from KVCachePool
 -> one-token cached decode
 -> write new token K/V into KVCachePool
 -> stream text
 -> scheduler finish/free path
```

The scheduler still owns:

* request queue draining
* slot admission
* active request tracking
* KV block metadata allocation
* KV block metadata expansion
* finish/fail policy
* metrics and snapshots

The custom Llama engine owns:

* prompt tokenization
* TinyLlama weight loading
* prompt prefill
* next-token selection
* writing real K/V tensors into `KVCachePool`
* gathering K/V from `KVCachePool`
* one-token cached decode
* tokenizer decoding for streamed text pieces

## What Lives in RequestState

`RequestState` stores request-local serving state.

In v2, it stores:

```python
request_state.input_ids
```

Full token sequence for the request, including prompt and generated tokens. This is used for bookkeeping, debugging, and parity checks.

```python
request_state.next_token
```

The next token selected by the previous prefill/decode step. The next `decode_step` consumes this token, emits it as text, and then computes the following token.

```python
request_state.block_table
```

The request-local logical-to-physical KV block mapping allocated by `KVBlockManager`.

```python
request_state.prompt_tokens
request_state.generated_tokens
request_state.num_computed_tokens
```

Token accounting used by the scheduler and metrics.

In v2, this should intentionally remain `None`:

```python
request_state.past_key_values = None
```

The K/V cache no longer lives on `RequestState`. The source of truth is `KVCachePool`.

## What Lives in KVBlockManager

`KVBlockManager` owns metadata, not tensors.

It tracks:

```text
request_id -> block_table
```

where `block_table` maps request-local logical blocks to global physical block IDs.

Example:

```text
block_size_tokens = 4
block_table = [0, 1]
```

means:

```text
request token positions 0..3 -> physical block 0
request token positions 4..7 -> physical block 1
```

The scheduler uses `KVBlockManager` to:

* allocate prompt KV blocks during admission
* ensure decode-time capacity before each token
* update `request_state.block_table`
* free request KV metadata when the request finishes or fails

## What Lives in KVCachePool

`KVCachePool` owns the physical K/V tensors.

Conceptual layout:

```text
key_cache[layer_id, physical_block_id, block_offset, kv_head, head_dim]
value_cache[layer_id, physical_block_id, block_offset, kv_head, head_dim]
```

For TinyLlama, the relevant dimensions are:

```text
num_layers = 22
num_kv_heads = 4
head_dim = 64
```

A single token’s K/V for one layer has shape:

```text
[num_kv_heads, head_dim]
```

A contiguous PyTorch cache for one layer has shape:

```text
[1, num_kv_heads, seq_len, head_dim]
```

The v2 transfer utilities bridge these formats.

## KV Transfer Utilities

The runtime utility module:

```text
runtime/kv_cache_transfer.py
```

contains:

```python
write_past_key_values_to_pool(...)
```

Writes full contiguous per-layer K/V into `KVCachePool`.

Used after prompt prefill.

```python
write_last_token_past_key_values_to_pool(...)
```

Writes only the newest token position from each layer’s present K/V tensors.

Used after one-token decode.

```python
gather_past_key_values_from_pool(...)
```

Reads request-local K/V from `KVCachePool` and reconstructs contiguous PyTorch `past_key_values`.

Used before v2 decode attention.

## v2 Init Path

During request admission, the scheduler allocates KV metadata first:

```text
prompt_tokens = decode_engine.count_prompt_tokens(prompt)
block_table = kv_block_manager.allocate_for_tokens(request_id, prompt_tokens)
request_state.block_table = block_table
decode_engine.init_request_state(request_state)
```

Then `CustomLlamaDecodeEngine.init_request_state` performs prefill:

```text
tokenize prompt
run custom cached model on prompt
produce prompt logits
produce per-layer prompt K/V
select first next_token
write prompt K/V into KVCachePool
clear request_state.past_key_values
store input_ids, next_token, prompt_tokens
```

Important invariant after init:

```text
request_state.input_ids length == prompt_tokens
request_state.generated_tokens == 0
request_state.next_token is not None
request_state.past_key_values is None
KVCachePool contains K/V for token positions [0, prompt_tokens)
```

## v2 Decode Path

Before every decode step, the scheduler ensures block capacity:

```text
token_position = prompt_tokens + generated_tokens
kv_block_manager.ensure_capacity_for_token(request_id, token_position)
request_state.block_table = kv_block_manager.get_block_tables(request_id)
```

Then the engine decodes:

```text
cache_seq_len = prompt_tokens + generated_tokens

gather past K/V from KVCachePool for positions [0, cache_seq_len)

consume request_state.next_token

append token to request_state.input_ids

run one-token cached decode using gathered K/V

write the newly produced K/V token into KVCachePool at token_position = cache_seq_len

select the next token for the following decode step

return text piece to scheduler
```

Important invariant after each decode step:

```text
request_state.input_ids length == prompt_tokens + generated_tokens
KVCachePool contains K/V for positions [0, prompt_tokens + generated_tokens)
request_state.past_key_values is None
```

## Validated Tests

### KVCachePool Writeback/Gather Roundtrip

Test:

```text
experiments/tests/llama/test_llama_kv_cache_pool_writeback_correctness.py
```

This validates:

```text
contiguous K/V
 -> write through block_table
 -> physical KVCachePool layout
 -> gather back
 -> contiguous K/V
```

Observed result:

```text
global_max_error=0.0
global_mean_error=0.0
```

### Decode Correctness Using KVCachePool-gathered Cache

Test:

```text
experiments/tests/llama/test_llama_kv_cache_pool_decode_correctness.py
```

This validates:

```text
cached decode using original contiguous past_key_values
==
cached decode using past_key_values gathered from KVCachePool
```

This proves that the physical KV layout can preserve real TinyLlama K/V tensors well enough to drive decode logits.

### CustomLlamaDecodeEngine KVCachePool Smoke Test

Test:

```text
experiments/tests/llama/test_custom_llama_kv_cache_pool_engine_smoke.py
```

This validates:

```text
CustomLlamaDecodeEngine
 -> prefill prompt
 -> write prompt K/V into KVCachePool
 -> keep request_state.past_key_values as None
 -> gather K/V for decode
 -> write generated-token K/V into KVCachePool
 -> emit real TinyLlama text
```

Expected generation:

```text
Paris.\n\n
```

### Scheduler Integration

The existing scheduler tests validate that v2 runs through:

```text
RequestQueue
 -> EngineScheduler
 -> KVBlockManager
 -> CustomLlamaDecodeEngine
 -> KVCachePool
 -> RequestState streaming
 -> finish/free path
```

Validated cases:

```text
single request
multi-request continuous-batching shape
```

For the two-request case:

```text
request_a_text='Paris.\n\n'
request_b_text='Berlin.\n\n'
decode_iterations=4
tokens_generated=8
```

## Current Limitations

The backend still gathers from `KVCachePool` into contiguous PyTorch tensors before attention.

Current v2 decode path:

```text
KVCachePool
 -> gather contiguous past_key_values
 -> PyTorch cached attention
 -> write newest K/V back into KVCachePool
```

Target v3 decode path:

```text
KVCachePool + block_table
 -> CUDA paged attention
 -> attention output
```

This means v2 validates the memory layout and lifecycle, but it does not yet get the performance benefit of paged attention.

Other current limitations:

* no CUDA paged attention in the real TinyLlama decode path yet
* no batched tensorized model execution across requests
* no custom CUDA kernels for RMSNorm, RoPE, or MLP
* no quantization
* no tensor parallelism
* no prefix caching

These are acceptable. The immediate goal is correctness and architecture, not full production parity.

## Accurate Project Claim After v2

The accurate claim is:

```text
Built an engine-agnostic LLM inference runtime with a custom TinyLlama backend that performs prompt prefill, stores real per-layer K/V tensors in a paged KVCachePool layout, gathers K/V for cached decode, streams real generated tokens through the scheduler, and frees KV metadata through the runtime lifecycle.
```

A shorter resume-style version:

```text
Implemented a custom TinyLlama inference backend with RMSNorm, RoPE, GQA attention, SwiGLU MLP, KV-cache writeback, paged KV metadata management, and scheduler-integrated cached decoding, validated against Hugging Face parity tests.
```

## Next Milestone: CUDA Paged Attention

The next version should replace the gather-then-PyTorch-attention path with CUDA paged attention during decode.

The next ladder is:

```text
v3a: inspect current CUDA paged attention API
v3b: adapt TinyLlama Q/K/V shapes to CUDA backend expectations
v3c: compare CUDA paged attention output against PyTorch attention for one layer
v3d: integrate CUDA paged attention into cached decode path
```

The first concrete files to inspect are:

```text
cuda_backend/paged_attention.cpp
cuda_backend/paged_attention_kernel.cu
runtime/attention_backend.py
```

Those define the interface between the runtime and the CUDA paged attention implementation.
