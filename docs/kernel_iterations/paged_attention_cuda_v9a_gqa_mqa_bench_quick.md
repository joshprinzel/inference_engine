# paged_attention_cuda_v9a_gqa_mqa_bench

## Goal

Benchmark the batched CUDA paged attention decode kernel across MHA, GQA, and MQA configurations.

This benchmark tests the performance effect of reducing KV heads while keeping query heads fixed.

## Environment

- Loaded extension: `/mnt/c/Users/joshp_ya/VSCodeProjects/Personal/portfolio/llm-inference-systems-lab/inference-server/cuda_backend/paged_attention_cuda.cpython-313-x86_64-linux-gnu.so`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- Device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Benchmark Config

- Profile: `quick`
- Attention configs: `[{'name': 'mha', 'num_query_heads': 16, 'num_kv_heads': 16, 'head_dim': 128}, {'name': 'gqa', 'num_query_heads': 16, 'num_kv_heads': 4, 'head_dim': 128}, {'name': 'mqa', 'num_query_heads': 16, 'num_kv_heads': 1, 'head_dim': 128}]`
- Batch sizes: `[1, 8, 32]`
- Sequence lengths: `[128, 512]`
- Number of cases: `18`
- Block size tokens: `8`
- Total blocks: `32768`
- Dtype: `float16`
- Warmup iterations: `10`
- Trials per case: `3`
- Reference timing enabled: `True`

## Results

| mode | batch | seq_len | q_heads | kv_heads | q/kv | CTAs | blocks/req | max_abs_diff | ref med ms | cuda med ms | cuda min ms | cuda max ms | speedup | req/ms | attended tok/ms | q-elems/ms | kv-elems/ms | iters | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mha | 1 | 128 | 16 | 16 | 1 | 16 | 16 | 0.00006104 | 1.303979 | 0.058146 | 0.058071 | 0.058218 | 22.43x | 17.20 | 2201.35 | 4508365.12 | 4508365.12 | 300 | True |
| mha | 8 | 128 | 16 | 16 | 1 | 128 | 16 | 0.00012207 | 10.266604 | 0.067502 | 0.067389 | 0.078602 | 152.09x | 118.51 | 15169.90 | 31067961.62 | 31067961.62 | 100 | True |
| mha | 32 | 128 | 16 | 16 | 1 | 512 | 16 | 0.00024414 | 40.469453 | 0.173066 | 0.172687 | 1.574451 | 233.84x | 184.90 | 23667.24 | 48470506.22 | 48470506.22 | 100 | True |
| mha | 1 | 512 | 16 | 16 | 1 | 16 | 64 | 0.00003052 | 3.975878 | 0.460291 | 0.221635 | 0.783698 | 8.64x | 2.17 | 1112.34 | 2278069.98 | 2278069.98 | 300 | True |
| mha | 8 | 512 | 16 | 16 | 1 | 128 | 64 | 0.00006104 | 28.422166 | 0.803594 | 0.581929 | 1.338399 | 35.37x | 9.96 | 5097.10 | 10438860.71 | 10438860.71 | 100 | True |
| mha | 32 | 512 | 16 | 16 | 1 | 512 | 64 | 0.00012207 | 111.901064 | 1.011364 | 1.010565 | 5.050901 | 110.64x | 31.64 | 16199.91 | 33177409.54 | 33177409.54 | 50 | True |
| gqa | 1 | 128 | 16 | 4 | 4 | 16 | 16 | 0.00000000 | 1.233138 | 0.045704 | 0.045647 | 0.046141 | 26.98x | 21.88 | 2800.62 | 5735677.07 | 1433919.27 | 300 | True |
| gqa | 8 | 128 | 16 | 4 | 4 | 128 | 16 | 0.00012207 | 8.400763 | 0.169277 | 0.168294 | 0.169482 | 49.63x | 47.26 | 6049.24 | 12388845.26 | 3097211.32 | 100 | True |
| gqa | 32 | 128 | 16 | 4 | 4 | 512 | 16 | 0.00012207 | 34.849187 | 0.718275 | 0.165765 | 1.482568 | 48.52x | 44.55 | 5702.55 | 11678832.50 | 2919708.13 | 100 | True |
| gqa | 1 | 512 | 16 | 4 | 4 | 16 | 64 | 0.00001526 | 3.617611 | 0.327748 | 0.220856 | 0.836847 | 11.04x | 3.05 | 1562.17 | 3199333.53 | 799833.38 | 300 | True |
| gqa | 8 | 512 | 16 | 4 | 4 | 128 | 64 | 0.00006104 | 29.531475 | 1.195674 | 0.251044 | 1.739663 | 24.70x | 6.69 | 3425.68 | 7015800.97 | 1753950.24 | 100 | True |
| gqa | 32 | 512 | 16 | 4 | 4 | 512 | 64 | 0.00012207 | 120.946650 | 0.628593 | 0.628593 | 3.622892 | 192.41x | 50.91 | 26064.57 | 53380249.25 | 13345062.31 | 50 | True |
| mqa | 1 | 128 | 16 | 1 | 16 | 16 | 16 | 0.00001526 | 1.485865 | 0.046568 | 0.045923 | 0.048640 | 31.91x | 21.47 | 2748.66 | 5629260.31 | 351828.77 | 300 | True |
| mqa | 8 | 128 | 16 | 1 | 16 | 128 | 16 | 0.00012207 | 11.162286 | 0.260362 | 0.259707 | 0.260731 | 42.87x | 30.73 | 3932.98 | 8054747.00 | 503421.69 | 100 | True |
| mqa | 32 | 128 | 16 | 1 | 16 | 512 | 16 | 0.00012207 | 41.748154 | 1.136005 | 0.165458 | 1.316997 | 36.75x | 28.17 | 3605.62 | 7384304.90 | 461519.06 | 100 | True |
| mqa | 1 | 512 | 16 | 1 | 16 | 16 | 64 | 0.00001526 | 4.003123 | 0.222092 | 0.221843 | 0.799928 | 18.02x | 4.50 | 2305.35 | 4721359.68 | 295084.98 | 300 | True |
| mqa | 8 | 512 | 16 | 1 | 16 | 128 | 64 | 0.00012207 | 30.911243 | 1.227571 | 0.254566 | 1.688853 | 25.18x | 6.52 | 3336.67 | 6833500.27 | 427093.77 | 100 | True |
| mqa | 32 | 512 | 16 | 1 | 16 | 512 | 64 | 0.00012207 | 115.785977 | 0.622100 | 0.619643 | 3.053322 | 186.12x | 51.44 | 26336.58 | 53937318.35 | 3371082.40 | 50 | True |

## Correctness

All benchmark cases passed correctness checks before timing.

## Timing Method

Each row reports the median of multiple CUDA-event timing trials. The minimum and maximum CUDA timings are included to expose benchmark variance.

## Interpretation

MHA uses one KV head per query head. GQA shares each KV head across multiple query heads. MQA shares one KV head across all query heads. Reducing KV heads reduces KV cache storage and changes the memory-access pattern, while the number of query heads still determines the number of sequence/head CTAs launched.

## Next Kernel Question

If GQA/MQA performance does not improve despite fewer KV heads, the bottleneck is likely not raw KV-cache footprint yet. The next optimization pass should use profiling to inspect scalar V loads, serial softmax denominator computation, and CTA-level occupancy.
