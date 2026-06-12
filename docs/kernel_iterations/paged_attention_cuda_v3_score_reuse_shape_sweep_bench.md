# paged_attention_cuda_v3_score_reuse_shape_sweep_bench

## Goal

Benchmark the CUDA paged attention decode kernel across a small matrix of sequence lengths, KV head counts, and head dimensions.

This benchmark is intended for kernel iteration guidance, not final production performance claims.

## Environment

- Loaded extension: `/mnt/c/Users/joshp_ya/VSCodeProjects/Personal/portfolio/llm-inference-systems-lab/inference-server/cuda_backend/paged_attention_cuda.cpython-313-x86_64-linux-gnu.so`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- Device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Benchmark Config

- Sequence lengths: `[64, 128, 256, 512]`
- Shape configs: `[{'num_kv_heads': 2, 'head_dim': 32}, {'num_kv_heads': 8, 'head_dim': 64}, {'num_kv_heads': 16, 'head_dim': 128}]`
- Block size tokens: `8`
- Total blocks: `512`
- Dtype: `float16`
- Default warmup iterations: `25`

## Results

| heads | head_dim | seq_len | blocks | max_abs_diff | reference ms | cuda ms | speedup | tok/ms | elems/ms | iters | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 32 | 64 | 8 | 0.00000381 | 0.631555 | 0.066651 | 9.48x | 960.22 | 61454.31 | 1000 | True |
| 2 | 32 | 128 | 16 | 0.00000000 | 0.992451 | 0.131140 | 7.57x | 976.05 | 62467.47 | 300 | True |
| 2 | 32 | 256 | 32 | 0.00000000 | 1.800144 | 0.259656 | 6.93x | 985.92 | 63098.95 | 300 | True |
| 2 | 32 | 512 | 64 | 0.00003052 | 3.205291 | 0.499644 | 6.42x | 1024.73 | 65582.73 | 300 | True |
| 8 | 64 | 64 | 8 | 0.00000095 | 0.668829 | 0.061153 | 10.94x | 1046.55 | 535833.91 | 300 | True |
| 8 | 64 | 128 | 16 | 0.00000000 | 1.045077 | 0.113630 | 9.20x | 1126.46 | 576749.80 | 300 | True |
| 8 | 64 | 256 | 32 | 0.00000000 | 2.050959 | 0.227157 | 9.03x | 1126.97 | 577009.75 | 300 | True |
| 8 | 64 | 512 | 64 | 0.00000000 | 3.407954 | 0.427981 | 7.96x | 1196.32 | 612513.45 | 100 | True |
| 16 | 128 | 64 | 8 | 0.00003052 | 0.640495 | 0.056183 | 11.40x | 1139.13 | 2332928.37 | 300 | True |
| 16 | 128 | 128 | 16 | 0.00000095 | 1.013565 | 0.108974 | 9.30x | 1174.59 | 2405562.97 | 100 | True |
| 16 | 128 | 256 | 32 | 0.00012207 | 1.793843 | 0.217969 | 8.23x | 1174.48 | 2405336.89 | 100 | True |
| 16 | 128 | 512 | 64 | 0.00006104 | 3.806751 | 0.428083 | 8.89x | 1196.03 | 2449467.82 | 100 | True |

## Correctness

All benchmark cases passed correctness checks before timing.

## Interpretation

The PyTorch reference is intentionally simple and includes Python-level looping over the paged KV cache. The CUDA kernel should increasingly benefit as sequence length, head count, and head dimension grow. This v3 kernel stores QK scores once in shared memory and reuses them for softmax denominator and value accumulation.

## Next Kernel Question

The next optimization question is whether v3 is limited by serial denominator computation, scalar V loads, one-CTA-per-head work granularity, or lack of batching across requests.
