# Runtime Benchmark Summary

Generated at: `2026-07-03T23:10:26`

Measured configurations: `1`
Measured rows: `3`

| backend | policy | requests | slots | new tokens | block size | repeats | tok/s median | tok/s min | tok/s max | TTFT ms median | latency ms median | backend ms median | backend ms p95 median | peak KV blocks | correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| custom-cuda-paged | fcfs | 4 | 4 | 8 | 16 | 3 | 72.29 | 62.57 | 96.55 | 220.627 | 442.447 | 31.641 | 37.113 | 4 | True |

## Notes

- Warmup rows are written to JSONL/CSV but excluded from this summary.
- Summary rows aggregate repeated measured runs by benchmark configuration.
- `total_wall_seconds` is end-to-end scheduler wall time.
- `backend_ms_*` comes from the decode engine's backend timing and includes Python/model/backend work inside `decode_step`.
- `kv_peak_used_blocks` is useful for graphing KV pressure under concurrency.
- `correctness_passed` checks generated text prefixes for the benchmark prompts.
- `avg_ttft_ms` is average request time-to-first-token for finished successful requests.
- `avg_latency_ms` is average end-to-end request latency for finished successful requests.
- `policy_name` identifies the scheduler policy used for admission/decode selection.
