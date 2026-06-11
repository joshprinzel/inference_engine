import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from paged_attention_reference import paged_attention_decode_reference

import paged_attention_cuda
print("loaded extension:", paged_attention_cuda.__file__)


def fill_request_kv(
    cache_pool: KVCachePool,
    layer_id: int,
    block_table: list[int],
    seq_len: int,
) -> None:
    for token_position in range(seq_len):
        key = torch.randn(
            cache_pool.layout.num_kv_heads,
            cache_pool.layout.head_dim,
            device=cache_pool.key_cache.device,
            dtype=cache_pool.key_cache.dtype,
        )

        value = torch.randn(
            cache_pool.layout.num_kv_heads,
            cache_pool.layout.head_dim,
            device=cache_pool.value_cache.device,
            dtype=cache_pool.value_cache.dtype,
        )

        cache_pool.write_request_token(
            layer_id=layer_id,
            block_table=block_table,
            token_position=token_position,
            key=key,
            value=value,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    torch.manual_seed(0)

    layout = KVCacheLayout(
        num_layers=2,
        total_blocks=16,
        block_size_tokens=8,
        num_kv_heads=2,
        head_dim=32,
        dtype="float16",
        device="cuda",
    )

    block_manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    layer_id = 0
    seq_len = 1
    num_query_heads = layout.num_kv_heads

    request_id = "req-0"
    block_table = block_manager.allocate_for_tokens(
        request_id=request_id,
        num_tokens=seq_len,
    )

    fill_request_kv(
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    q = torch.randn(
        num_query_heads,
        layout.head_dim,
        device="cuda",
        dtype=layout.torch_dtype,
    )

    block_table_tensor = torch.tensor(
        block_table,
        dtype=torch.int32,
        device="cuda",
    )

    reference_output = paged_attention_decode_reference(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    cuda_output = paged_attention_cuda.paged_attention_decode(
        q,
        cache_pool.key_cache,
        cache_pool.value_cache,
        block_table_tensor,
        layer_id,
        seq_len,
    )

    print("reference finite:", torch.isfinite(reference_output).all().item())
    print("cuda finite:", torch.isfinite(cuda_output).all().item())
    print("reference has nan:", torch.isnan(reference_output).any().item())
    print("cuda has nan:", torch.isnan(cuda_output).any().item())
    print("cuda output sample:", cuda_output.flatten()[:8])
    print("reference output sample:", reference_output.flatten()[:8])

    diff = cuda_output.float() - reference_output.float()
    print("diff finite:", torch.isfinite(diff).all().item())
    print("diff sample:", diff.flatten()[:8])

    max_abs_diff = diff.abs().max().item()

    print("paged attention cuda v1")
    print("---")
    print("layout:", layout.snapshot())
    print("block_table:", block_table)
    print("reference_output_shape:", tuple(reference_output.shape))
    print("cuda_output_shape:", tuple(cuda_output.shape))
    print("max_abs_diff:", max_abs_diff)

    assert max_abs_diff < 1e-2


if __name__ == "__main__":
    main()