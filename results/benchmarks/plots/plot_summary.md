# Runtime Benchmark Plots

Source CSV: `results/benchmarks/runtime_benchmark_20260627_205209.csv`

Aggregated measured configurations: `9`

## Generated plots

- `results/benchmarks/plots/tokens_per_second_vs_requests.png`
- `results/benchmarks/plots/backend_ms_median_vs_requests.png`
- `results/benchmarks/plots/backend_ms_p95_vs_requests.png`
- `results/benchmarks/plots/kv_peak_blocks_vs_requests.png`

## Plot interpretation checklist

- `tokens_per_second_vs_requests.png`: shows end-to-end serving throughput as active requests increase.
- `backend_ms_median_vs_requests.png`: shows median decode engine latency and exposes per-request loop scaling.
- `backend_ms_p95_vs_requests.png`: shows tail latency behavior.
- `kv_peak_blocks_vs_requests.png`: shows KV cache pressure under concurrency.
- Block-size sensitivity plots show how block granularity affects throughput and KV allocation.
