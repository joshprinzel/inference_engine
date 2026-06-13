# paged_attention_cuda_v5_batched Validation

## Goal

Validate the batched CUDA paged attention decode kernel against the single-request PyTorch paged attention reference.

## Kernel Scope

- Decode-only attention
- Batched requests
- Variable sequence lengths per batch
- One CUDA block per sequence/head pair
- `num_query_heads == num_kv_heads`
- FP16 inputs with FP32 accumulation
- Paged KV layout using padded physical block tables

## Environment

- Loaded extension: `/mnt/c/Users/joshp_ya/VSCodeProjects/Personal/portfolio/llm-inference-systems-lab/inference-server/cuda_backend/paged_attention_cuda.cpython-313-x86_64-linux-gnu.so`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- Device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Layout

- num_layers: `2`
- total_blocks: `4096`
- block_size_tokens: `8`
- num_kv_heads: `2`
- head_dim: `32`
- dtype: `float16`

## Correctness Grid

| batch_size | seq_lens | block_tables_shape | max_abs_diff | passed |
|---:|---|---|---:|---|
| 1 | `[1]` | `(1, 1)` | 0.00000000 | True |
| 2 | `[7, 8]` | `(2, 1)` | 0.00000000 | True |
| 4 | `[9, 19, 32, 64]` | `(4, 8)` | 0.00000000 | True |
| 4 | `[64, 128, 256, 512]` | `(4, 64)` | 0.00000000 | True |

## Result

All batched correctness cases passed.

## Interpretation

Passing this grid validates that the CUDA kernel can map each batch row to its own padded block table, use its own sequence length, and produce the same output as independently decoding each request with the reference path.

## Next Step

The next step is a batch-size sweep benchmark to verify that more sequence/head CTAs improve GPU occupancy and serving-shaped throughput.
