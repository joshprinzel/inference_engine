# Runtime Benchmark Summary

Generated at: `2026-06-27T20:52:34`

| backend | requests | slots | new tokens | block size | tok/s | backend ms median | backend ms p95 | peak KV blocks | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| custom-cuda-paged | 1 | 1 | 8 | 16 | 10.94 | 20.514 | 51.576 | 1 | True |
| custom-cuda-paged | 1 | 1 | 16 | 16 | 47.52 | 18.016 | 22.179 | 2 | True |
| custom-cuda-paged | 1 | 1 | 32 | 16 | 45.02 | 19.657 | 25.305 | 3 | True |
| custom-cuda-paged | 2 | 2 | 8 | 16 | 38.90 | 40.820 | 48.732 | 2 | True |
| custom-cuda-paged | 2 | 2 | 16 | 16 | 40.06 | 42.274 | 50.419 | 4 | True |
| custom-cuda-paged | 2 | 2 | 32 | 16 | 43.94 | 41.535 | 51.878 | 6 | True |
| custom-cuda-paged | 4 | 4 | 8 | 16 | 40.95 | 80.401 | 87.525 | 4 | True |
| custom-cuda-paged | 4 | 4 | 16 | 16 | 43.38 | 81.877 | 87.076 | 8 | True |
| custom-cuda-paged | 4 | 4 | 32 | 16 | 48.09 | 77.967 | 84.971 | 12 | True |

## Notes

- `total_wall_seconds` is end-to-end scheduler wall time.
- `backend_ms_*` comes from the decode engine's backend timing and includes Python/model/backend work inside `decode_step`.
- `kv_peak_used_blocks` is useful for graphing KV pressure under concurrency.
- `correctness_passed` checks generated text prefixes for the benchmark prompts.
