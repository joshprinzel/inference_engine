# Runtime Benchmark Summary

Generated at: `2026-06-27T22:37:19`

Measured configurations: `12`
Measured rows: `36`

| backend | requests | slots | new tokens | block size | repeats | tok/s median | tok/s min | tok/s max | backend ms median | backend ms p95 median | peak KV blocks | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| custom-cuda-paged | 4 | 4 | 8 | 4 | 3 | 106.16 | 105.46 | 108.18 | 21.075 | 25.907 | 16 | True |
| custom-cuda-paged | 4 | 4 | 8 | 8 | 3 | 110.36 | 105.84 | 112.61 | 20.110 | 24.002 | 8 | True |
| custom-cuda-paged | 4 | 4 | 8 | 16 | 3 | 104.76 | 99.48 | 110.91 | 21.407 | 25.757 | 4 | True |
| custom-cuda-paged | 4 | 4 | 8 | 32 | 3 | 81.15 | 79.96 | 103.80 | 27.984 | 38.009 | 4 | True |
| custom-cuda-paged | 4 | 4 | 16 | 4 | 3 | 120.60 | 119.28 | 126.62 | 21.045 | 28.524 | 24 | True |
| custom-cuda-paged | 4 | 4 | 16 | 8 | 3 | 124.56 | 114.77 | 126.77 | 21.575 | 29.470 | 12 | True |
| custom-cuda-paged | 4 | 4 | 16 | 16 | 3 | 127.80 | 114.69 | 132.31 | 22.585 | 25.272 | 8 | True |
| custom-cuda-paged | 4 | 4 | 16 | 32 | 3 | 120.38 | 78.47 | 127.67 | 22.323 | 29.713 | 4 | True |
| custom-cuda-paged | 4 | 4 | 32 | 4 | 3 | 89.01 | 84.26 | 137.72 | 40.870 | 44.124 | 40 | True |
| custom-cuda-paged | 4 | 4 | 32 | 8 | 3 | 86.74 | 77.41 | 102.80 | 39.847 | 44.255 | 20 | True |
| custom-cuda-paged | 4 | 4 | 32 | 16 | 3 | 154.45 | 138.92 | 156.16 | 20.929 | 26.262 | 12 | True |
| custom-cuda-paged | 4 | 4 | 32 | 32 | 3 | 146.73 | 145.84 | 155.67 | 22.508 | 28.349 | 8 | True |

## Notes

- Warmup rows are written to JSONL/CSV but excluded from this summary.
- Summary rows aggregate repeated measured runs by benchmark configuration.
- `total_wall_seconds` is end-to-end scheduler wall time.
- `backend_ms_*` comes from the decode engine's backend timing and includes Python/model/backend work inside `decode_step`.
- `kv_peak_used_blocks` is useful for graphing KV pressure under concurrency.
- `correctness_passed` checks generated text prefixes for the benchmark prompts.
