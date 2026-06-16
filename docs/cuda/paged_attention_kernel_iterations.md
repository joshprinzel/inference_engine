# Paged Attention CUDA Kernel Iteration Map

Project: `llm-inference-systems-lab`  
Component: `inference-server/cuda_backend/`  
Purpose: interview-season review map for how the paged attention kernel evolved from correctness-first code into a profiler-driven CUDA backend.

---

## 0. Why This Kernel Exists

The serving engine lowers active decode requests into a `DecodeBatch`:

- `request_ids`
- `input_token_ids`
- `positions`
- `seq_lens`
- `block_tables`

The CUDA backend consumes the lowered batch and performs decode-time paged attention.

For decode attention, each request has one query token. For each sequence/query-head pair, the kernel computes:

```text
scores[token] = q · k[token]
prob[token]   = softmax(scores)[token]
out[d]        = Σ prob[token] * v[token, d]
```

Paged attention matters because the KV cache is not stored as one contiguous `[seq_len]` allocation per request. It is stored in fixed-size physical blocks, and each request has a logical-to-physical `block_table`.

The kernel's job is therefore:

```text
logical token position
    -> logical KV block id
    -> physical KV block id from block_table
    -> block offset
    -> K/V address
```

---

## 1. ABI and Tensor Layout

The batched kernel API uses:

```text
q            [batch, num_query_heads, head_dim]
key_cache    [layers, blocks, block_size, num_kv_heads, head_dim]
value_cache  [layers, blocks, block_size, num_kv_heads, head_dim]
block_tables [batch, max_blocks_per_seq]
seq_lens     [batch]
output       [batch, num_query_heads, head_dim]
```

The launch geometry is:

```text
grid.x = num_query_heads
grid.y = batch_size
block.x = THREADS_PER_HEAD = 128
```

So each CTA owns exactly one:

```text
(sequence_id, query_head_id)
```

This is simple, debuggable, and aligns naturally with the scheduler's decode batch abstraction.

---

## 2. v1 — Single-Request Correctness Kernel

### Goal

Prove the memory layout and paged addressing were correct.

### Design

- Single request.
- One block per attention head.
- Mostly serial work inside a CTA.
- Correctness-first, performance ignored.

### What It Taught

The first milestone was not speed. It was verifying that the kernel could correctly traverse:

```text
block_table -> physical KV block -> K/V vector
```

### Limitation

Too little parallelism. One or very few threads did most of the useful work.

---

## 3. v2 — Threaded Per-Head Kernel

### Goal

Parallelize dot products across `head_dim`.

### Design

- Still single request.
- One CTA per head.
- 128 threads per head.
- Threads cooperate over `head_dim`.
- Use block-wide reductions for QK score computation.

### Mechanism

For each token:

```text
for d = tid; d < head_dim; d += 128:
    partial += q[d] * k[token, d]

score = block_reduce_sum(partial)
```

### What It Fixed

QK dot product work was now distributed across the CTA instead of being mostly serial.

### Limitation

Still single-request only, so occupancy was poor for realistic serving workloads.

---

## 4. v3 — Score Reuse Kernel

### Goal

Avoid recomputing QK scores multiple times.

### Design

- Compute each QK score once.
- Store scores in shared memory:

```cpp
__shared__ float scores[MAX_SEQ_LEN];
```

- Reuse scores for:
  - max computation
  - softmax denominator
  - V accumulation

### Why This Helped

Before score reuse, the kernel could recompute `q · k[token]` for multiple softmax phases. v3 cut redundant dot-product work.

### Conceptual Structure

```text
Pass 1: compute scores[token]
Pass 2: compute denom from scores
Pass 3: accumulate V using scores
```

### Limitation

Still single request. Kernel launch had too few CTAs to fill the GPU.

---

## 5. v4 — Batched Decode Kernel

### Goal

Make the CUDA kernel match the serving engine's `DecodeBatch` abstraction.

### Design

New batched shape:

```text
q            [batch, num_heads, head_dim]
key_cache    [layers, blocks, block_size, num_heads, head_dim]
value_cache  [layers, blocks, block_size, num_heads, head_dim]
block_tables [batch, max_blocks]
seq_lens     [batch]
output       [batch, num_heads, head_dim]
```

Grid:

```text
blockIdx.x = head_id
blockIdx.y = sequence_id
```

### What It Fixed

The number of CTAs became:

```text
batch_size * num_heads
```

This directly fixed under-occupancy for decode workloads.

### Key Lesson

Batching is not just a scheduler feature. It changes the GPU occupancy profile by increasing independent CTA work.

---

## 6. v5 — GQA/MQA Support

### Goal

Support modern LLM attention layouts:

- MHA: `num_query_heads == num_kv_heads`
- GQA: `num_query_heads > num_kv_heads`
- MQA: `num_kv_heads == 1`

### Design

The semantic API changed to separate query heads from KV heads:

```text
q            [batch, num_query_heads, head_dim]
key_cache    [layers, blocks, block_size, num_kv_heads, head_dim]
value_cache  [layers, blocks, block_size, num_kv_heads, head_dim]
output       [batch, num_query_heads, head_dim]
```

Mapping:

```cpp
query_heads_per_kv_head = num_query_heads / num_kv_heads;
kv_head_id = query_head_id / query_heads_per_kv_head;
```

### What It Fixed

The kernel could now run MHA, GQA, and MQA correctly.

### Important Performance Lesson

GQA/MQA reduce KV memory footprint, but this kernel still launches one CTA per query head:

```text
CTAs = batch_size * num_query_heads
```

So GQA/MQA do not automatically produce huge speedups in this design. The output work and CTA count are still query-head driven.

---

## 7. v6a — Parallel Softmax Denominator

### Problem

v5 had a serial denominator computation:

```cpp
if(tid == 0){
    for token_pos in seq_len:
        denom += expf(scores[token_pos] - max_score);
}
```

That meant 127 threads sat idle while one thread did O(seq_len) work.

### Fix

Parallelize denominator computation:

```cpp
float denom_partial = 0.0f;

for(token_pos = tid; token_pos < seq_len; token_pos += THREADS_PER_HEAD){
    denom_partial += expf(scores[token_pos] - max_score);
}

float denom = block_reduce_sum(denom_partial);
```

### Result

Canonical GQA profile case:

```text
batch_size      = 32
seq_len         = 512
num_query_heads = 16
num_kv_heads    = 4
head_dim        = 128
```

Observed improvement over v5:

```text
v5  NSYS avg:    ~1.398 ms
v6a NSYS avg:    ~1.283 ms

v5  NSYS median: ~1.285 ms
v6a NSYS median: ~1.209 ms
```

Approximate improvement:

```text
avg:    ~8.2%
median: ~5.9%
```

### Profiler Diagnosis After v6a

Nsight Compute showed the next issue was not the denominator anymore. The QK pass still did one CTA-wide reduction per token.

For `seq_len=512`, the old structure implied thousands of CTA-wide synchronization points inside QK score generation.

---

## 8. v7 — Warp-Level QK Score Generation

### Problem

v6a still computed each token score like this:

```text
for each token:
    all 128 threads cooperate on q · k[token]
    block_reduce_sum(partial_score)
    store scores[token]
```

The block-wide reduction used shared memory and `__syncthreads()`. Paying that cost once per token was the dominant structural bottleneck.

### Goal

Keep the external ABI and one-CTA-per-sequence/query-head mapping, but remove the block-wide reduction-per-token pattern.

### Design

Use 4 warps per CTA:

```text
THREADS_PER_HEAD = 128
WARPS_PER_BLOCK  = 4
```

Each warp computes one token score:

```text
warp 0 -> token_base + 0
warp 1 -> token_base + 1
warp 2 -> token_base + 2
warp 3 -> token_base + 3
```

Each lane handles dimensions:

```text
lane 0 -> d = 0, 32, 64, 96
lane 1 -> d = 1, 33, 65, 97
...
```

Use warp shuffle reduction:

```cpp
__device__ float warp_reduce_sum(float value){
    for(int offset = 16; offset > 0; offset >>= 1){
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}
```

### v7 Structure

```text
Pass 1a: warp-level QK writes scores[token]
Pass 1b: block-wide max reduction over scores
Pass 2:  parallel denominator reduction over scores
Pass 3:  V accumulation using scores
```

### Correctness Bug Found

The first v7 version missed a required barrier between score writes and score reads:

```cpp
if(lane_id == 0){
    scores[token_pos] = score;
}

// Required before max reduction reads scores[]
__syncthreads();
```

Without this barrier, some threads could read stale shared-memory scores before all warp writers had finished.

This was a race, not a classic deadlock.

### Result

Fixed v7 canonical profile:

```text
GQA, batch=32, seq_len=512, q_heads=16, kv_heads=4, head_dim=128
```

Nsight Systems:

```text
avg kernel time:    ~662.5 us
median kernel time: ~723.7 us
```

Nsight Compute:

```text
duration:              ~938.9 us
compute throughput:     61.82%
memory throughput:      26.97%
DRAM throughput:        14.38%
No Eligible:            35.09%
Issued Warp/Scheduler:   0.65
Eligible Warps/Scheduler: 2.44
Active Warps/Scheduler:  8.83
```

Compared to v6a:

```text
v6a NCU duration: ~1.97 ms
v7  NCU duration: ~0.939 ms

improvement: ~52%
```

### Interpretation

v7 was a major structural improvement. It reduced synchronization pressure in QK and improved scheduler eligibility.

The kernel was still not DRAM-bandwidth bound:

```text
compute throughput >> DRAM throughput
```

So the next bottleneck was redundant passes, control overhead, and the materialized-score softmax structure.

---

## 9. v8 — Online Softmax Target

### Motivation

v7 still materializes:

```cpp
scores[MAX_SEQ_LEN]
```

and performs multiple sequence passes:

```text
1. compute scores
2. reduce max over scores
3. reduce denominator over scores
4. reread scores to accumulate V
```

v8 targets this structure.

### What “Online” Means Here

Online does not mean continuous batching.

In this context, online means:

```text
streaming / incremental softmax computation
```

Instead of storing every score and softmaxing later, the kernel updates the softmax state as each token score arrives.

### Online Softmax Recurrence

Maintain:

```text
m      = running max score
l      = running denominator
acc[d] = running weighted V accumulator
```

For each token:

```text
score = q · k[token]

m_new = max(m, score)
alpha = exp(m - m_new)
beta  = exp(score - m_new)

acc[d] = acc[d] * alpha + beta * V[token, d]
l      = l * alpha + beta
m      = m_new
```

Final:

```text
out[d] = acc[d] / l
```

### Data Movement Saved

For decode attention, the avoided intermediate is not an `N x N` matrix. Since decode has one query token, the intermediate score object is:

```text
scores[seq_len]
```

So v8 tries to remove:

```text
- shared-memory score vector
- separate max pass
- separate denominator pass
- rereading scores during V accumulation
```

### Relationship to FlashAttention

The principle is the same as FlashAttention:

```text
Naive/materialized attention:
    compute scores/probs as an intermediate
    then multiply by V

FlashAttention-style attention:
    stream through K/V tiles
    maintain online softmax state
    avoid materializing attention scores/probs
```

For prefill, the avoided object is approximately:

```text
[query_len, key_len]
```

or self-attention:

```text
[N, N]
```

For decode, where `query_len = 1`, the avoided object is:

```text
[1, seq_len]
```

### Expected Tradeoff

v8 is not guaranteed to beat v7 immediately.

It removes redundant passes and score storage, but introduces:

```text
- recurrence dependency through m/l/acc
- tighter coupling between QK and V
- tile synchronization
- more sequential softmax state updates
```

So v8 is a real experiment:

```text
Does reducing score materialization and full-sequence passes beat the added online recurrence dependencies?
```

---

## 10. Kernel Evolution Summary Table

| Version | Main Change | Bottleneck Addressed | New Limitation Exposed |
|---|---|---|---|
| v1 | Single-request correctness kernel | Addressing correctness | Too serial |
| v2 | Threaded per-head dot products | Dot-product parallelism | Still single request |
| v3 | Shared-memory score reuse | Redundant QK recomputation | Too few CTAs |
| v4 | Batched decode kernel | Under-occupancy | No GQA/MQA |
| v5 | GQA/MQA support | Modern attention layouts | Serial denominator |
| v6a | Parallel denominator | One-thread O(seq_len) softmax denominator | QK block reduction per token |
| v7 | Warp-level QK reductions | CTA-wide reduction per token | Materialized scores and multi-pass softmax |
| v8 | Online softmax target | Score storage and repeated score passes | Recurrence dependency / tile design |

---

## 11. Interview Explanation: Short Version

The project started with a correctness-first paged attention kernel, then evolved through profiler-driven bottleneck removal.

First, I validated paged KV addressing with a single-request implementation. Then I parallelized per-head dot products across a CTA. After that, I stored QK scores in shared memory to avoid recomputing them. The next bottleneck was GPU occupancy, so I generalized the kernel to batched decode with one CTA per sequence/query-head pair. Then I added GQA/MQA support by mapping query heads to shared KV heads.

Profiling showed the softmax denominator was serial in v5, so v6a parallelized it across the CTA. After that, Nsight showed the main problem was the QK pass: every token paid a CTA-wide reduction and synchronization. v7 fixed that by assigning token score generation to warps and using `__shfl_down_sync` reductions. That roughly halved the canonical GQA b32 s512 kernel duration under Nsight Compute.

The next step, v8, is to remove the materialized score vector and implement online softmax. That moves the kernel toward a FlashAttention-style streaming design where QK scores are consumed as they are produced instead of stored and reread through multiple passes.

---

## 12. Interview Explanation: Deep Version

The key design decision is that the current batched decode kernel maps one CTA to one `(sequence_id, query_head_id)`. That makes the kernel simple and aligns with the serving scheduler's `DecodeBatch`, but it also means CTA count is controlled by query heads, not KV heads. This is why GQA/MQA reduce KV memory footprint but do not automatically reduce CTA count or output work.

The biggest profiler-driven improvement was v7. In v6a, QK score generation looked parallel at first glance because 128 threads cooperated on each dot product. But the implementation required a block-wide reduction for every token. For `seq_len=512`, that meant hundreds of CTA-wide reductions and synchronization points. v7 changed the granularity: each of the 4 warps in the CTA computes a different token score using warp shuffle reductions. This removes most of the block-wide synchronization from QK and only keeps block-wide reductions for max and denominator.

The profiler validated the hypothesis. `No Eligible` dropped, `Eligible Warps/Scheduler` increased, and NCU duration dropped from roughly 1.97 ms to roughly 0.94 ms on the canonical GQA b32 s512 case. The kernel remained compute/control limited rather than DRAM-bandwidth bound, which suggested the next target should be the materialized softmax structure rather than memory coalescing alone.

v8 targets that by replacing the score vector with online softmax. Instead of computing all scores, reducing max, reducing denominator, and then doing V accumulation, v8 maintains running `(m, l, acc)` state and folds each token's V contribution into the output accumulator as soon as the QK score is available.

---

## 13. Things To Remember Before Resuming Work

1. v7 should remain as the stable baseline.
2. v8 should be added as a new kernel, not overwrite v7 immediately.
3. The benchmark must compare v7 and v8 on the same canonical case.
4. Correctness tolerance may shift slightly in v8 because online softmax changes accumulation order.
5. GQA/MQA speedups are limited until the kernel reuses KV work across query heads.
6. Raw profiler reports should not be committed; keep markdown summaries in `docs/kernel_iterations/`.

---

## 14. Canonical Benchmark / Profile Case

Use this shape for apples-to-apples profiling:

```text
mode:              gqa
batch_size:        32
seq_len:           512
num_query_heads:   16
num_kv_heads:      4
head_dim:          128
```

This shape has enough CTAs and sequence length to expose the real kernel behavior.

---

## 15. Next Planned Work

### v8

Implement online softmax decode while preserving:

```text
- same ABI
- same tensor layouts
- same CTA mapping
- same GQA/MQA support
```

### v9

If v8 is promising, optimize tile structure:

```text
- fewer barriers
- better V load structure
- vectorized K/V loads
- head_dim=128 specialization
- possibly half2/float4 paths
```

### v10

Separate raw CUDA launch code from the PyTorch extension:

```text
- shared launch function
- PyTorch binding calls launch function
- native CMake profiler can call same launch function
```

### Later Integration

Connect the CUDA backend into the real serving path:

```text
ModelExecutor / scheduler DecodeBatch
    -> CUDA paged attention op
    -> real serving decode step
```

Longer term, this feeds into a graph/runtime layer inspired by vLLM-style execution, CUDA Graphs, and compiler-backed runtimes.
