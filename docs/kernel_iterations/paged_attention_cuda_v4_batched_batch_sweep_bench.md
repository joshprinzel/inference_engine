# paged_attention_cuda_v4_batched_batch_sweep_bench

## Goal

Benchmark the batched CUDA paged attention decode kernel across increasing batch sizes and sequence lengths.

This benchmark tests whether exposing more sequence/head CTAs improves GPU occupancy and serving-shaped throughput.

## Environment

- Loaded extension: `/mnt/c/Users/joshp_ya/VSCodeProjects/Personal/portfolio/llm-inference-systems-lab/inference-server/cuda_backend/paged_attention_cuda.cpython-313-x86_64-linux-gnu.so`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- Device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Benchmark Config

- Batch sizes: `[1, 2, 4, 8, 16, 32]`
- Sequence lengths: `[128, 256, 512]`
- Num KV heads: `16`
- Head dim: `128`
- Block size tokens: `8`
- Total blocks: `32768`
- Dtype: `float16`
- Default warmup iterations: `25`
- Trials per case: `5`

## Results

| batch | seq_len | CTAs | blocks/req | max_abs_diff | ref median ms | cuda median ms | cuda min ms | cuda max ms | speedup | req/ms | attended tok/ms | elems/ms | iters | passed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 128 | 16 | 16 | 0.00006104 | 1.282956 | 0.127300 | 0.099113 | 0.130765 | 10.08x | 7.86 | 1005.50 | 2059257.34 | 300 | True |
| 2 | 128 | 32 | 16 | 0.00012207 | 2.631233 | 0.264329 | 0.107674 | 0.436265 | 9.95x | 7.57 | 968.49 | 1983471.05 | 300 | True |
| 4 | 128 | 64 | 16 | 0.00006104 | 5.066011 | 0.135117 | 0.134021 | 0.829303 | 37.49x | 29.60 | 3789.31 | 7760515.00 | 300 | True |
| 8 | 128 | 128 | 16 | 0.00012207 | 9.977365 | 0.153201 | 0.151460 | 1.190922 | 65.13x | 52.22 | 6684.05 | 13688924.90 | 100 | True |
| 16 | 128 | 256 | 16 | 0.00024414 | 16.034437 | 0.227052 | 0.226335 | 1.506662 | 70.62x | 70.47 | 9019.98 | 18472917.10 | 100 | True |
| 32 | 128 | 512 | 16 | 0.00012207 | 33.323992 | 0.381194 | 0.336507 | 1.666642 | 87.42x | 83.95 | 10745.18 | 22006125.38 | 100 | True |
| 1 | 256 | 16 | 32 | 0.00003052 | 1.964165 | 0.256246 | 0.199270 | 0.777390 | 7.67x | 3.90 | 999.04 | 2046035.86 | 300 | True |
| 2 | 256 | 32 | 32 | 0.00003052 | 4.269059 | 0.248139 | 0.197171 | 0.950624 | 17.20x | 8.06 | 2063.36 | 4225759.05 | 300 | True |
| 4 | 256 | 64 | 32 | 0.00012207 | 8.205168 | 0.274627 | 0.261868 | 1.507010 | 29.88x | 14.57 | 3728.70 | 7636377.19 | 100 | True |
| 8 | 256 | 128 | 32 | 0.00012207 | 16.113193 | 0.294482 | 0.293939 | 1.418127 | 54.72x | 27.17 | 6954.59 | 14242992.97 | 100 | True |
| 16 | 256 | 256 | 32 | 0.00012207 | 32.502681 | 0.449229 | 0.354509 | 1.796004 | 72.35x | 35.62 | 9117.85 | 18673353.81 | 100 | True |
| 32 | 256 | 512 | 32 | 0.00012207 | 64.194746 | 0.887624 | 0.726999 | 1.837589 | 72.32x | 36.05 | 9229.14 | 18901271.47 | 50 | True |
| 1 | 512 | 16 | 64 | 0.00006104 | 3.893142 | 0.390369 | 0.386669 | 0.806356 | 9.97x | 2.56 | 1311.58 | 2686112.95 | 300 | True |
| 2 | 512 | 32 | 64 | 0.00001526 | 7.093566 | 0.515553 | 0.412037 | 1.898158 | 13.76x | 3.88 | 1986.22 | 4067769.65 | 100 | True |
| 4 | 512 | 64 | 64 | 0.00006104 | 13.835684 | 0.514724 | 0.400783 | 2.027889 | 26.88x | 7.77 | 3978.83 | 8148648.96 | 100 | True |
| 8 | 512 | 128 | 64 | 0.00012207 | 27.532483 | 0.582851 | 0.454574 | 1.911163 | 47.24x | 13.73 | 7027.53 | 14392381.89 | 100 | True |
| 16 | 512 | 256 | 64 | 0.00012207 | 55.084873 | 0.939377 | 0.751882 | 1.849016 | 58.64x | 17.03 | 8720.68 | 17859946.65 | 50 | True |
| 32 | 512 | 512 | 64 | 0.00012207 | 114.568545 | 1.391411 | 1.386230 | 1.709240 | 82.34x | 23.00 | 11775.10 | 24115397.10 | 50 | True |

## Correctness

All benchmark cases passed correctness checks before timing.

## Timing Method

Each row reports the median of multiple CUDA-event timing trials. The minimum and maximum CUDA timings are included to expose benchmark variance.

## Interpretation

The v4 kernel launches one CUDA block per sequence/head pair. Therefore, the number of CTAs scales as `batch_size * num_heads`. If the single-request kernel was under-occupying the GPU, throughput should improve as batch size increases.

## Next Kernel Question

If throughput saturates at moderate batch size, the next bottlenecks are likely scalar memory access, serial softmax denominator computation, or inefficient per-head CTA work decomposition.
