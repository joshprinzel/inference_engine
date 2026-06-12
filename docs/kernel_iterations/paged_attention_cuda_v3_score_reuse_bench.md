# paged_attention_cuda_v3_score_reuse_bench

## Goal

Benchmark the CUDA paged attention decode kernel against the PyTorch reference implementation across increasing sequence lengths.

This benchmark is intended for iteration guidance, not final production performance claims.

## Environment

- Loaded extension: `/mnt/c/Users/joshp_ya/VSCodeProjects/Personal/portfolio/llm-inference-systems-lab/inference-server/cuda_backend/paged_attention_cuda.cpython-313-x86_64-linux-gnu.so`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- Device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Benchmark Config

- Warmup iterations: `50`
- Benchmark iterations: `500`
- Block size tokens: `8`
- Total blocks: `512`
- Num KV heads: `2`
- Head dim: `32`
- Dtype: `float16`

## Layout

- num_layers: `2`
- total_blocks: `512`
- block_size_tokens: `8`
- num_kv_heads: `2`
- head_dim: `32`
- dtype: `float16`

## Results

| seq_len | blocks | max_abs_diff | reference ms | cuda ms | speedup | passed |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.00000000 | 0.479840 | 0.024979 | 19.21x | True |
| 8 | 1 | 0.00000000 | 0.533699 | 0.018010 | 29.63x | True |
| 19 | 3 | 0.00000000 | 0.570935 | 0.025336 | 22.53x | True |
| 64 | 8 | 0.00000000 | 0.762597 | 0.066966 | 11.39x | True |
| 128 | 16 | 0.00000000 | 1.244019 | 0.133136 | 9.34x | True |
| 256 | 32 | 0.00000000 | 2.236414 | 0.259140 | 8.63x | True |
| 512 | 64 | 0.00000000 | 3.920982 | 0.652800 | 6.01x | True |

## Correctness

All benchmark cases passed correctness checks before timing.

## Interpretation

The PyTorch reference is intentionally simple and includes Python-level looping over the paged KV cache. The CUDA kernel should increasingly benefit as sequence length grows, but this v2 kernel still recomputes QK scores multiple times and is not expected to represent final performance.

## Next Kernel Question

If v2 is not significantly faster, the likely bottleneck is repeated QK score recomputation. The next iteration should consider storing scores or using an online softmax structure.
