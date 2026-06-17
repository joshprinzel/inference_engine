# paged_attention_cuda_v2 Validation

## Goal

Validate that the CUDA C++ paged attention decode kernel matches the PyTorch paged attention reference across important sequence-length cases.

This test is correctness-focused, not performance-focused.

## Kernel Scope

- Decode-only attention
- Single request
- One CUDA block per attention head
- `num_query_heads == num_kv_heads`
- FP16 inputs with FP32 accumulation
- Paged KV layout using physical block tables

## Environment

- Loaded extension: `/mnt/c/Users/joshp_ya/VSCodeProjects/Personal/portfolio/llm-inference-systems-lab/inference-server/cuda_backend/paged_attention_cuda.cpython-313-x86_64-linux-gnu.so`
- PyTorch: `2.12.0+cu130`
- CUDA: `13.0`
- Device: `NVIDIA GeForce RTX 4070 Laptop GPU`

## Layout

- num_layers: `2`
- total_blocks: `16`
- block_size_tokens: `8`
- num_kv_heads: `2`
- head_dim: `32`
- dtype: `float16`

## Correctness Grid

| seq_len | block_table | max_abs_diff | passed |
|---:|---|---:|---|
| 1 | `[0]` | 0.00000000 | True |
| 7 | `[0]` | 0.00000000 | True |
| 8 | `[0]` | 0.00000000 | True |
| 9 | `[0, 1]` | 0.00000000 | True |
| 19 | `[0, 1, 2]` | 0.00000000 | True |

## Result

All correctness cases passed.

## Why These Sequence Lengths Matter

- `1`: one-token attention; output should equal the only value vector
- `7`: partial first block
- `8`: exact block boundary
- `9`: crosses from block 0 into block 1
- `19`: multi-block sequence

## Interpretation

The CUDA kernel consumes the same ABI as the Python reference: query tensor, physical key/value cache tensors, block table, layer id, and sequence length.

Passing this grid validates that the kernel can walk the paged KV block table and reproduce dense/reference attention results across block-boundary cases.

## Next Step

If this is v2, the next step is either benchmarking against v1 or moving toward a more efficient kernel that avoids repeated QK recomputation.
