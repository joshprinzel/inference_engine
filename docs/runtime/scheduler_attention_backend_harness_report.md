# Scheduler Attention Backend Harness Report

## Source

```text
results/scheduler_attention_backend_harness.csv
```

## Summary

| Metric | Value |
|---|---:|
| Backend | `reference` |
| Decode steps | 8 |
| Total tokens emitted | 32 |
| Backend median latency | 4.960296 ms |
| Backend min latency | 4.364768 ms |
| Backend p95 latency | 118.153554 ms |
| Backend max latency | 178.472922 ms |
| Average active batch size | 4.00 |
| Max active batch size | 4 |
| Initial KV utilization | 1.0000 |
| Max KV utilization | 1.0000 |
| Final KV utilization | 0.0000 |

## Interpretation

This report summarizes the scheduler-owned synthetic decode harness.

The harness validates this runtime path:

```text
RequestState
  -> KVBlockManager
  -> DecodeBatch
  -> KVCachePool
  -> AttentionBackend
  -> CUDA paged attention kernel
```

This is not full model execution. Query tensors, generated token IDs, and generated K/V entries are synthetic.
The purpose is to validate scheduler/KV/backend plumbing and measure the attention backend inside a repeated decode loop.

## Per-Step Tail

Last 10 decode steps:

```text
 decode_step  active_batch_size   backend  backend_ms  kv_used_blocks  kv_free_blocks  kv_utilization  total_tokens_emitted
           0                  4 reference  178.472922              68               0             1.0                     4
           1                  4 reference    4.668805              68               0             1.0                     8
           2                  4 reference    4.364768              68               0             1.0                    12
           3                  4 reference    6.131871              68               0             1.0                    16
           4                  4 reference    5.056051              68               0             1.0                    20
           5                  4 reference    4.581187              68               0             1.0                    24
           6                  4 reference    5.838669              68               0             1.0                    28
           7                  4 reference    4.864541               0              68             0.0                    32
```
