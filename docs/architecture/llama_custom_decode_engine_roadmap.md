# Llama Custom Decode Engine Roadmap

## Goal

Build a real Llama-style custom decode engine that plugs into the existing `DecodeEngine` interface.

The final target is:

```text
EngineScheduler
    -> CustomLlamaDecodeEngine
    -> real model weights
    -> real Q/K/V projections
    -> real KVCachePool writeback
    -> DecodeBatch lowering
    -> custom CUDA paged attention
    -> real logits
    -> sampling
    -> streamed output
```

This turns the project from an engine-agnostic runtime with a synthetic CUDA path into a small vLLM-style inference runtime with a real model execution path.

---

## Non-Goals

Do not support arbitrary Hugging Face models initially.

Do not optimize before correctness.

Do not implement training.

Do not implement distributed inference.

Do not implement speculative decoding.

Do not build a complex frontend before the model path works.

Do not claim parity with vLLM.

---

## Current Foundation

Already implemented:

```text
EngineScheduler
DecodeEngine interface
HFDecodeEngine
SyntheticCudaDecodeEngine
KVBlockManager
KVCachePool
DecodeBatch
AttentionBackend
CUDA paged attention backend
server backend switch
benchmark reports
```

The current CUDA path is synthetic at the model-math level.

The next milestone is replacing synthetic model math with real Llama-style model execution.

---

## Target Architecture

```text
RequestQueue
    -> EngineScheduler
    -> CustomLlamaDecodeEngine
        -> tokenizer
        -> embeddings
        -> transformer layers
            -> RMSNorm
            -> Q/K/V projection
            -> RoPE
            -> KVCachePool writeback
            -> CUDA paged attention
            -> output projection
            -> residual
            -> RMSNorm
            -> SwiGLU MLP
            -> residual
        -> final norm
        -> lm_head
        -> sampling
        -> streamed token
```

---

## Phase 1: Llama Math Correctness

Purpose:

Build confidence in the model components before involving the scheduler or CUDA backend.

Milestones:

```text
1. Inspect model config and weight names.
2. Implement RMSNorm and compare against Hugging Face.
3. Implement RoPE and compare against Hugging Face.
4. Implement Q/K/V projection and head reshaping.
5. Implement GQA repeat/mapping semantics.
6. Implement attention reference in PyTorch.
7. Implement SwiGLU MLP.
8. Compare one full transformer block against Hugging Face.
9. Compare full model logits against Hugging Face for one prompt.
```

Exit condition:

```text
Custom PyTorch Llama forward produces logits close to Hugging Face for a fixed prompt.
```

---

## Phase 2: CustomLlamaDecodeEngine, PyTorch Attention

Purpose:

Plug real model execution into the existing `DecodeEngine` interface before using custom CUDA attention.

Milestones:

```text
1. Implement count_prompt_tokens.
2. Implement init_request_state with real tokenization and prefill.
3. Implement decode_step for one request.
4. Use greedy decoding only.
5. Stream real generated text through EngineScheduler.
6. Compare generated tokens against Hugging Face under deterministic settings.
```

Exit condition:

```text
EngineScheduler -> CustomLlamaDecodeEngine can generate real text for one request.
```

---

## Phase 3: KVCachePool Integration

Purpose:

Move K/V ownership from temporary PyTorch tensors into project-owned paged KV cache.

Milestones:

```text
1. During prefill, write real K/V tensors into KVCachePool.
2. During decode, write the new token K/V into KVCachePool.
3. Validate token_position -> physical_block mapping.
4. Compare gathered KV from KVCachePool against reference contiguous KV.
5. Preserve correctness for single-request decode.
```

Exit condition:

```text
CustomLlamaDecodeEngine owns real paged KV storage.
```

---

## Phase 4: CUDA Paged Attention Integration

Purpose:

Replace PyTorch decode attention with the existing CUDA paged attention backend.

Milestones:

```text
1. Build DecodeBatch from active request states.
2. Feed real Q tensors into AttentionBackend.
3. Use KVCachePool as real K/V source.
4. Compare CUDA attention output against PyTorch attention reference.
5. Integrate CUDA attention output into transformer block.
6. Generate correct next tokens for one request.
```

Exit condition:

```text
Single-request Llama decode uses custom CUDA paged attention.
```

---

## Phase 5: Batched Decode

Purpose:

Use the existing scheduler to run multiple real Llama requests together.

Milestones:

```text
1. Support multiple active requests.
2. Build batched Q tensors.
3. Build batched DecodeBatch.
4. Handle ragged sequence lengths.
5. Sample per request.
6. Stream outputs per request.
7. Free KV blocks on finish.
```

Exit condition:

```text
EngineScheduler can continuously batch real Llama decode requests through custom CUDA attention.
```

---

## Phase 6: Dashboard

Purpose:

Make the system understandable to non-technical interviewers.

Dashboard should show:

```text
selected backend
active requests
waiting requests
finished requests
tokens/sec
decode step count
KV blocks used/free
KV utilization
last backend ms
DecodeBatch size
per-request generated tokens
request lifecycle timeline
```

Optional visualizations:

```text
KV block table
DecodeBatch table
tokens/sec chart
backend latency chart
request slot view
```

Exit condition:

```text
A recruiter can understand what the system is doing without reading code.
```

---

## Correctness Strategy

For every model component, compare against Hugging Face.

Use fixed seeds.

Use deterministic greedy decoding.

Use small prompts.

Compare:

```text
intermediate tensors
attention outputs
block outputs
final logits
next token IDs
generated text
```

Tolerance should be documented per dtype.

---

## Benchmark Strategy

Benchmark stages separately:

```text
HFDecodeEngine baseline
SyntheticCudaDecodeEngine runtime/backend path
CustomLlamaDecodeEngine PyTorch attention
CustomLlamaDecodeEngine CUDA attention
```

Measure:

```text
tokens/sec
time to first token
decode latency
backend attention ms
KV utilization
batch size scaling
```

---

## Final Resume Claim

Target claim:

> Built a small vLLM-style Llama inference runtime from scratch with continuous batching, paged KV cache management, DecodeBatch lowering, and a custom CUDA paged-attention backend integrated into a real Llama-style decoder engine.

---

## Honest Limitations

This project will not claim production parity with vLLM.

It will not claim to support arbitrary models.

It will not claim optimized prefill.

It will not claim distributed inference.

The strength is educational depth and end-to-end systems implementation.