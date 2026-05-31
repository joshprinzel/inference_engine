# Static Microbatch Validation

## Goal

Validate that the real inference server performs actual batched GPU execution, not just request-level scheduling.

The test compares completed engine batches of size 1, 2, and 4 using the static microbatch scheduler.

## Setup

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Endpoint: `/generate_stream`
- Scheduler: static microbatch engine
- Max batch size: 4
- Output tokens per request: 32
- Prompt token targets: 32 and 128
- Repeats: 3
- Warmup: first batch excluded from summary

## Batch-Level Results

| batch_size | batches | avg_prefill | avg_decode | p95_decode | avg_total | avg_batch_tok_s | p50_batch_tok_s | p95_batch_tok_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 0.0425 | 0.8948 | 0.9820 | 0.9400 | 36.02 | 37.13 | 38.92 |
| 2 | 6 | 0.0425 | 0.8784 | 0.8950 | 0.9264 | 72.91 | 72.71 | 74.19 |
| 4 | 6 | 0.0477 | 0.8970 | 0.9429 | 0.9590 | 143.00 | 145.10 | 148.04 |

## Key Finding

Decode wall time stays nearly flat as batch size increases:

| batch_size | avg_decode |
|---|---:|
| 1 | 0.8948s |
| 2 | 0.8784s |
| 4 | 0.8970s |

But each decode step produces one token per active request, so total tokens generated per batch scales with batch size:

| batch_size | decode_steps | tokens_generated |
|---|---:|---:|
| 1 | 32 | 32 |
| 2 | 32 | 64 |
| 4 | 32 | 128 |

This yields near-linear batch throughput scaling:

| batch_size | avg_batch_tok_s | speedup_vs_batch_1 |
|---|---:|---:|
| 1 | 36.02 | 1.00x |
| 2 | 72.91 | 2.02x |
| 4 | 143.00 | 3.97x |

## Interpretation

Static microbatching successfully converts concurrent requests into real batched GPU work.

The scheduler admits multiple waiting requests into a fixed active batch, runs one batched prefill, and then performs one batched decode forward per generated token step. Each request still receives streamed output independently.

The near-flat decode wall time from batch size 1 to 4, combined with nearly 4x token throughput at batch size 4, validates that this implementation is doing actual batched decode rather than serially looping over requests.

## Current Limitation

This is static microbatching, not full continuous batching.

A batch must finish before the next batch is admitted. The next major scheduler milestone is continuous admission, where new requests can join while existing requests are already decoding.

## Reproduction

Run the benchmark:

```bash
python experiments/benchmark_streaming.py \
  --output-path results/benchmark_streaming_static_microbatch_batch_metrics.csv \
  --batch-metrics-output-path results/benchmark_streaming_static_microbatch_engine_batches.csv
```
