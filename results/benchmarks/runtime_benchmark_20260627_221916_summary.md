# Runtime Benchmark Summary

Generated at: `2026-06-27T22:20:40`

| backend | requests | slots | new tokens | block size | tok/s | backend ms median | backend ms p95 | peak KV blocks | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| custom-cuda-paged | 1 | 1 | 8 | 16 | 47.52 | 17.025 | 17.388 | 1 | True |
| custom-cuda-paged | 1 | 1 | 8 | 16 | 43.01 | 17.167 | 21.935 | 1 | True |
| custom-cuda-paged | 1 | 1 | 8 | 16 | 42.83 | 18.073 | 22.839 | 1 | True |
| custom-cuda-paged | 1 | 1 | 16 | 16 | 46.37 | 19.137 | 26.332 | 2 | True |
| custom-cuda-paged | 1 | 1 | 16 | 16 | 50.00 | 16.685 | 20.380 | 2 | True |
| custom-cuda-paged | 1 | 1 | 16 | 16 | 50.57 | 17.123 | 17.976 | 2 | True |
| custom-cuda-paged | 1 | 1 | 32 | 16 | 48.14 | 17.975 | 27.355 | 3 | True |
| custom-cuda-paged | 1 | 1 | 32 | 16 | 42.81 | 21.125 | 33.091 | 3 | True |
| custom-cuda-paged | 1 | 1 | 32 | 16 | 53.47 | 17.145 | 19.740 | 3 | True |
| custom-cuda-paged | 2 | 2 | 8 | 16 | 68.71 | 18.375 | 21.455 | 2 | True |
| custom-cuda-paged | 2 | 2 | 8 | 16 | 73.32 | 17.821 | 20.118 | 2 | True |
| custom-cuda-paged | 2 | 2 | 8 | 16 | 64.88 | 21.765 | 25.599 | 2 | True |
| custom-cuda-paged | 2 | 2 | 16 | 16 | 69.62 | 22.402 | 41.115 | 4 | True |
| custom-cuda-paged | 2 | 2 | 16 | 16 | 85.20 | 18.541 | 19.776 | 4 | True |
| custom-cuda-paged | 2 | 2 | 16 | 16 | 88.41 | 18.085 | 21.145 | 4 | True |
| custom-cuda-paged | 2 | 2 | 32 | 16 | 80.63 | 21.467 | 26.480 | 6 | True |
| custom-cuda-paged | 2 | 2 | 32 | 16 | 75.07 | 23.909 | 30.306 | 6 | True |
| custom-cuda-paged | 2 | 2 | 32 | 16 | 81.32 | 22.013 | 27.073 | 6 | True |
| custom-cuda-paged | 4 | 4 | 8 | 16 | 86.77 | 29.117 | 31.722 | 4 | True |
| custom-cuda-paged | 4 | 4 | 8 | 16 | 90.92 | 22.691 | 30.604 | 4 | True |
| custom-cuda-paged | 4 | 4 | 8 | 16 | 94.27 | 25.231 | 29.114 | 4 | True |
| custom-cuda-paged | 4 | 4 | 16 | 16 | 121.44 | 23.212 | 30.595 | 8 | True |
| custom-cuda-paged | 4 | 4 | 16 | 16 | 128.18 | 22.012 | 24.904 | 8 | True |
| custom-cuda-paged | 4 | 4 | 16 | 16 | 117.07 | 23.727 | 29.489 | 8 | True |
| custom-cuda-paged | 4 | 4 | 32 | 16 | 90.91 | 40.640 | 42.690 | 12 | True |
| custom-cuda-paged | 4 | 4 | 32 | 16 | 113.85 | 26.181 | 41.714 | 12 | True |
| custom-cuda-paged | 4 | 4 | 32 | 16 | 86.74 | 41.549 | 44.406 | 12 | True |

## Notes

- `total_wall_seconds` is end-to-end scheduler wall time.
- `backend_ms_*` comes from the decode engine's backend timing and includes Python/model/backend work inside `decode_step`.
- `kv_peak_used_blocks` is useful for graphing KV pressure under concurrency.
- `correctness_passed` checks generated text prefixes for the benchmark prompts.
