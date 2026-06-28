# Runtime Benchmark Summary

Generated at: `2026-06-27T22:27:52`

Measured configurations: `9`
Measured rows: `27`

| backend | requests | slots | new tokens | block size | repeats | tok/s median | tok/s min | tok/s max | backend ms median | backend ms p95 median | peak KV blocks | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| custom-cuda-paged | 1 | 1 | 8 | 16 | 3 | 40.13 | 28.65 | 42.76 | 18.433 | 23.792 | 1 | True |
| custom-cuda-paged | 1 | 1 | 16 | 16 | 3 | 34.71 | 25.75 | 38.85 | 23.890 | 35.404 | 2 | True |
| custom-cuda-paged | 1 | 1 | 32 | 16 | 3 | 32.08 | 28.34 | 40.97 | 32.369 | 35.301 | 3 | True |
| custom-cuda-paged | 2 | 2 | 8 | 16 | 3 | 45.38 | 44.75 | 53.26 | 29.594 | 37.931 | 2 | True |
| custom-cuda-paged | 2 | 2 | 16 | 16 | 3 | 85.20 | 51.79 | 89.41 | 18.065 | 21.199 | 4 | True |
| custom-cuda-paged | 2 | 2 | 32 | 16 | 3 | 94.61 | 92.13 | 94.75 | 18.280 | 22.019 | 6 | True |
| custom-cuda-paged | 4 | 4 | 8 | 16 | 3 | 101.76 | 101.68 | 113.79 | 20.693 | 25.922 | 4 | True |
| custom-cuda-paged | 4 | 4 | 16 | 16 | 3 | 136.36 | 135.87 | 137.21 | 20.566 | 22.843 | 8 | True |
| custom-cuda-paged | 4 | 4 | 32 | 16 | 3 | 152.14 | 148.01 | 156.73 | 21.173 | 25.035 | 12 | True |

## Notes

- Warmup rows are written to JSONL/CSV but excluded from this summary.
- Summary rows aggregate repeated measured runs by benchmark configuration.
- `total_wall_seconds` is end-to-end scheduler wall time.
- `backend_ms_*` comes from the decode engine's backend timing and includes Python/model/backend work inside `decode_step`.
- `kv_peak_used_blocks` is useful for graphing KV pressure under concurrency.
- `correctness_passed` checks generated text prefixes for the benchmark prompts.
