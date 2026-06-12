# paged_attention_cuda_v2_bench

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
| 1 | 1 | 0.00000000 | 0.330398 | 0.024971 | 13.23x | True |
| 8 | 1 | 0.00000000 | 0.327422 | 0.021203 | 15.44x | True |
| 19 | 3 | 0.00000000 | 0.348396 | 0.039477 | 8.83x | True |
| 64 | 8 | 0.00000000 | 0.687995 | 0.125827 | 5.47x | True |
| 128 | 16 | 0.00000000 | 1.031485 | 0.239860 | 4.30x | True |
| 256 | 32 | 0.00000000 | 1.878043 | 0.397967 | 4.72x | True |

## Correctness

All benchmark cases passed correctness checks before timing.

## Interpretation

The PyTorch reference is intentionally simple and includes Python-level looping over the paged KV cache. The CUDA kernel should increasingly benefit as sequence length grows, but this v2 kernel still recomputes QK scores multiple times and is not expected to represent final performance.

## Next Kernel Question

If v2 is not significantly faster, the likely bottleneck is repeated QK score recomputation. The next iteration should consider storing scores or using an online softmax structure.
