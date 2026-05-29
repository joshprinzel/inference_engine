## Real Model Benchmark: Prefill vs Decode

Model:

- Qwen/Qwen2.5-0.5B-Instruct
- single-request execution
- greedy decoding
- Hugging Face Transformers
- measured with direct `ModelRunner.prefill()` and `ModelRunner.decode_from_prefill()`

Observation:

For this small model, prefill time is much smaller than decode time across the tested prompt/output lengths. Decode dominates total latency because each generated token requires an autoregressive model step using the KV cache. This benchmark provides the first real timing data that can later be used to calibrate the simulator’s prefill/decode cost model.

## Benchmark: Prefill vs Decode

Model:

- Qwen/Qwen2.5-0.5B-Instruct
- Hugging Face Transformers
- single-request execution
- greedy decoding
- direct `ModelRunner.prefill()` and `ModelRunner.decode_from_prefill()`
- results saved to `results/benchmark_prefill_decode.csv`

| target prompt tokens | actual prompt tokens | max new tokens | avg prefill | p95 prefill | avg decode | p95 decode | avg total | p95 total | decode tok/s | total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 41 | 32 | 0.0494 | 0.0523 | 1.0645 | 1.1192 | 1.1138 | 1.1715 | 30.15 | 28.80 |
| 32 | 41 | 64 | 0.0560 | 0.0650 | 1.9201 | 2.0903 | 1.9761 | 2.1449 | 33.72 | 32.75 |
| 32 | 41 | 128 | 0.0512 | 0.0566 | 3.9157 | 4.2190 | 3.9669 | 4.2561 | 33.19 | 32.76 |
| 128 | 161 | 32 | 0.0479 | 0.0524 | 1.0339 | 1.0637 | 1.0819 | 1.1161 | 30.99 | 29.60 |
| 128 | 161 | 64 | 0.0468 | 0.0498 | 1.6466 | 1.6652 | 1.6934 | 1.7117 | 39.30 | 38.17 |
| 128 | 161 | 128 | 0.0492 | 0.0558 | 2.9513 | 3.0286 | 3.0005 | 3.0903 | 43.46 | 42.75 |
| 512 | 641 | 32 | 0.0459 | 0.0526 | 0.7731 | 0.8037 | 0.8191 | 0.8604 | 41.76 | 39.44 |
| 512 | 641 | 64 | 0.0439 | 0.0479 | 1.4115 | 1.4275 | 1.4555 | 1.4765 | 45.38 | 44.01 |
| 512 | 641 | 128 | 0.0456 | 0.0522 | 2.8408 | 2.8604 | 2.8863 | 2.8996 | 45.06 | 44.35 |
| 1024 | 1281 | 32 | 0.0586 | 0.0608 | 0.7579 | 0.7873 | 0.8165 | 0.8463 | 42.31 | 39.28 |
| 1024 | 1281 | 64 | 0.0629 | 0.0657 | 1.4550 | 1.4575 | 1.5179 | 1.5233 | 44.01 | 42.19 |
| 1024 | 1281 | 128 | 0.0651 | 0.0684 | 3.2468 | 3.5561 | 3.3119 | 3.6192 | 40.50 | 39.69 |

Observation:

For this small model, prefill time is much smaller than decode time across the tested prompt/output lengths. Total latency is dominated by autoregressive decoding, especially as `max_new_tokens` increases. These measurements provide real timing data that can later calibrate the simulator’s prefill/decode cost model.


Scheduler-backed streaming baseline:
- Single model-worker ownership removes destructive concurrent model.generate() contention.
- Concurrency now appears primarily as queueing delay.
- At concurrency 4, scheduler-backed serving improves average latency from ~7.5s to ~2.2s versus naive request-level streaming.
- TTFT increases because requests wait their turn in the queue; this is expected for a serialized scheduler.
- This establishes the control-plane foundation needed for continuous batching.


implementation | model ownership | decode ownership | concurrency behavior
--- | --- | --- | ---
naive streaming | request thread | HF generate | model contention
scheduler streaming | scheduler | HF generate | serialized queueing
manual engine | scheduler | manual decode loop | engine-visible queueing



## Next Milestone Continuous Batching

### Intermediate Step: Static Microbatching

waiting queue
   ↓
admit up to max_batch_size
   ↓
batch prefill
   ↓
batch decode step
   ↓
stream one token per request
   ↓
repeat until whole batch finishes
   ↓
admit next batch