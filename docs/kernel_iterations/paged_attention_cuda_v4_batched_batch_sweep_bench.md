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
| 1 | 128 | 16 | 16 | 0.00006104 | 1.195418 | 0.125918 | 0.102700 | 0.127840 | 9.49x | 7.94 | 1016.54 | 2081865.00 | 300 | True |
| 2 | 128 | 32 | 16 | 0.00012207 | 2.207109 | 0.129649 | 0.129260 | 0.130150 | 17.02x | 15.43 | 1974.57 | 4043914.35 | 300 | True |
| 4 | 128 | 64 | 16 | 0.00006104 | 5.111535 | 0.134393 | 0.133458 | 0.704406 | 38.03x | 29.76 | 3809.72 | 7802301.22 | 300 | True |
| 8 | 128 | 128 | 16 | 0.00012207 | 10.204713 | 0.152033 | 0.151593 | 1.229384 | 67.12x | 52.62 | 6735.37 | 13794032.34 | 100 | True |
| 16 | 128 | 256 | 16 | 0.00024414 | 19.636111 | 0.225505 | 0.224881 | 1.354148 | 87.08x | 70.95 | 9081.83 | 18599582.59 | 100 | True |
| 32 | 128 | 512 | 16 | 0.00012207 | 39.461191 | 0.381686 | 0.380180 | 1.643172 | 103.39x | 83.84 | 10731.34 | 21977786.54 | 100 | True |
| 1 | 256 | 16 | 32 | 0.00003052 | 1.858860 | 0.247859 | 0.198847 | 0.815203 | 7.50x | 4.03 | 1032.84 | 2115265.51 | 300 | True |
| 2 | 256 | 32 | 32 | 0.00003052 | 3.776259 | 0.252553 | 0.197635 | 1.042319 | 14.95x | 7.92 | 2027.30 | 4151912.29 | 300 | True |
| 4 | 256 | 64 | 32 | 0.00012207 | 7.660104 | 0.264018 | 0.263260 | 1.741322 | 29.01x | 15.15 | 3878.52 | 7943218.24 | 100 | True |
| 8 | 256 | 128 | 32 | 0.00012207 | 14.648453 | 0.294963 | 0.294359 | 1.294080 | 49.66x | 27.12 | 6943.24 | 14219753.17 | 100 | True |
| 16 | 256 | 256 | 32 | 0.00012207 | 29.845669 | 0.445809 | 0.444959 | 2.009733 | 66.95x | 35.89 | 9187.80 | 18816611.56 | 100 | True |
| 32 | 256 | 512 | 32 | 0.00012207 | 61.767617 | 0.849347 | 0.691958 | 0.851210 | 72.72x | 37.68 | 9645.06 | 19753086.83 | 50 | True |
| 1 | 512 | 16 | 64 | 0.00006104 | 3.793613 | 0.388820 | 0.388751 | 0.692361 | 9.76x | 2.57 | 1316.81 | 2696818.57 | 300 | True |
| 2 | 512 | 32 | 64 | 0.00001526 | 7.387638 | 0.498289 | 0.424059 | 2.329252 | 14.83x | 4.01 | 2055.03 | 4208709.14 | 100 | True |
| 4 | 512 | 64 | 64 | 0.00006104 | 14.100582 | 0.517652 | 0.402432 | 1.670154 | 27.24x | 7.73 | 3956.32 | 8102547.97 | 100 | True |
| 8 | 512 | 128 | 64 | 0.00012207 | 28.786616 | 0.586107 | 0.453038 | 1.927670 | 49.11x | 13.65 | 6988.49 | 14312420.47 | 100 | True |
| 16 | 512 | 256 | 64 | 0.00012207 | 53.448193 | 0.960840 | 0.781373 | 4.105134 | 55.63x | 16.65 | 8525.88 | 17460993.91 | 50 | True |
| 32 | 512 | 512 | 64 | 0.00012207 | 117.599844 | 1.488384 | 1.459077 | 2.527334 | 79.01x | 21.50 | 11007.91 | 22544204.53 | 50 | True |

## Correctness

All benchmark cases passed correctness checks before timing.

## Timing Method

Each row reports the median of multiple CUDA-event timing trials. The minimum and maximum CUDA timings are included to expose benchmark variance.

## Interpretation

The v4 kernel launches one CUDA block per sequence/head pair. Therefore, the number of CTAs scales as `batch_size * num_heads`. If the single-request kernel was under-occupying the GPU, throughput should improve as batch size increases.

## Next Kernel Question

If throughput saturates at moderate batch size, the next bottlenecks are likely scalar memory access, serial softmax denominator computation, or inefficient per-head CTA work decomposition.
