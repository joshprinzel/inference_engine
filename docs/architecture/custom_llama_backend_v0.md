# Custom Llama Backend v0: Full-Recompute Decode

## Goal

This backend proves that the engine-agnostic scheduler can drive a custom Llama-style model implementation, not just Hugging Face generation or a synthetic CUDA backend.

The v0 backend is correctness-first. It recomputes the full token sequence on every decode step and does not yet use KV-cached decode or CUDA paged attention.

## What It Uses

The backend uses Hugging Face for:

* loading TinyLlama weights
* tokenization
* tokenizer decoding
* rotary embedding table generation

The backend does not call:

* `model.forward`
* `model.generate`
* decoder layer forward methods
* attention module forward methods

The model path uses custom implementations for:

* token embedding lookup
* RMSNorm
* Q/K/V projection
* RoPE application
* grouped-query attention
* output projection
* SwiGLU MLP
* decoder-layer residual structure
* final RMSNorm
* lm_head projection
* greedy next-token selection

## Runtime Integration

The backend implements the same `DecodeEngine` protocol as the other engines:

* `count_prompt_tokens(prompt)`
* `init_request_state(request_state)`
* `decode_step(request_states, kv_block_manager)`

The scheduler still owns:

* request queue draining
* slot admission
* active request tracking
* KV block metadata reservation/freeing
* metrics
* finish/fail policy

The custom backend owns:

* prompt tokenization
* request-local token state
* full custom model forward
* next-token selection

In v0, request token state is stored in:

```python
request_state.input_ids
```

Each decode step:

```text
for each runnable request:
    run custom full forward on request_state.input_ids
    take argmax of final-position logits
    append next token to request_state.input_ids
    increment request_state.generated_tokens
    return text piece to scheduler
```

## Validated Tests

### Full Model Logits Parity

The custom full forward path matches Hugging Face logits closely enough to produce the same greedy next token.

### Stepwise Greedy Generation Parity

For the prompt:

```text
The capital of France is
```

custom generation and Hugging Face stepwise greedy generation produced the same token IDs:

```text
[3681, 29889, 13, 13, 29906, 29889, 350, 29889]
```

Decoded output:

```text
The capital of France is Paris.

2. B.
```

### Single-Request Scheduler Integration

A single request runs through:

```text
RequestQueue
 -> EngineScheduler
 -> CustomLlamaDecodeEngine
 -> RequestState streaming
 -> finish/free path
```

Observed output:

```text
generated_text='Paris.\n\n'
generated_tokens=4
decode_iterations=4
tokens_generated=4
```

### Multi-Request Scheduler Integration

Two requests run through the scheduler together:

```text
request_a_text='Paris.\n\n'
request_b_text='Berlin.\n\n'
request_a_tokens=4
request_b_tokens=4
decode_iterations=4
tokens_generated=8
```

Both requests receive separate KV metadata block tables during admission and all KV metadata is freed after completion.

## Current Limitations

This backend does not yet use:

* prefill/decode separation
* KV cache writeback
* `KVCachePool`
* CUDA paged attention
* batched tensor execution across requests

Although `decode_step` accepts multiple requests, v0 loops over them independently and recomputes each full sequence from scratch.

## Next Milestone

The next backend version should split full recompute into explicit prefill/decode phases.

Target:

```text
CustomLlamaDecodeEngine v1
    prefill prompt
    produce first logits
    store per-layer K/V
    decode one token using cached K/V
```

After v1 works with contiguous PyTorch K/V cache, the next milestone is writing K/V into the runtime `KVCachePool` layout:

```text
[layer_id, physical_block_id, block_offset, kv_head, head_dim]
```

Then CUDA paged attention can replace the PyTorch attention path during decode.

## Version Ladder

```text
v0: full recompute custom Llama backend        ✅
v1: contiguous PyTorch KV cache decode         ✅
v2: write K/V into KVCachePool                 next
v3: CUDA paged attention in real decode path
```

Do not jump directly from v0 to paged CUDA. The missing bridge is cached decode correctness.
