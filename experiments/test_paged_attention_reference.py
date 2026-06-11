import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from paged_attention_reference import (
    gather_kv_for_sequence,
    paged_attention_decode_batch_reference,
    paged_attention_decode_reference,
)


def dense_attention_reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Dense reference for one sequence.

    q:
        [num_query_heads, head_dim]

    keys/values:
        [seq_len, num_query_heads, head_dim]
    """

    scale = 1.0 / (q.shape[-1] ** 0.5)

    scores = torch.einsum("hd,shd->hs", q.float(), keys.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("hs,shd->hd", probs, values.float())

    return output.to(dtype=q.dtype)


def fill_request_kv(
    cache_pool: KVCachePool,
    layer_id: int,
    block_table: list[int],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fills one request's paged KV cache with random K/V and returns dense K/V.
    """

    keys = []
    values = []

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

        keys.append(key)
        values.append(value)

    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def main() -> None:
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    layout = KVCacheLayout(
        num_layers=2,
        total_blocks=16,
        block_size_tokens=8,
        num_kv_heads=2,
        head_dim=32,
        dtype="float16",
        device=device,
    )

    block_manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    layer_id = 0
    seq_len = 19
    num_query_heads = 2

    request_id = "req-0"
    block_table = block_manager.allocate_for_tokens(
        request_id=request_id,
        num_tokens=seq_len,
    )

    dense_keys, dense_values = fill_request_kv(
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    gathered_keys, gathered_values = gather_kv_for_sequence(
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    q = torch.randn(
        num_query_heads,
        layout.head_dim,
        device=device,
        dtype=layout.torch_dtype,
    )

    paged_output = paged_attention_decode_reference(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    dense_output = dense_attention_reference(
        q=q,
        keys=dense_keys,
        values=dense_values,
    )

    max_abs_diff = (paged_output.float() - dense_output.float()).abs().max().item()

    print("single sequence")
    print("---")
    print("layout:", layout.snapshot())
    print("block_table:", block_table)
    print("gathered_keys_match:", torch.equal(gathered_keys, dense_keys))
    print("gathered_values_match:", torch.equal(gathered_values, dense_values))
    print("paged_output_shape:", tuple(paged_output.shape))
    print("dense_output_shape:", tuple(dense_output.shape))
    print("max_abs_diff:", max_abs_diff)

    assert torch.equal(gathered_keys, dense_keys)
    assert torch.equal(gathered_values, dense_values)
    assert max_abs_diff < 1e-2

    # Batch test with two requests.
    request_id_1 = "req-1"
    seq_len_1 = 11

    block_table_1 = block_manager.allocate_for_tokens(
        request_id=request_id_1,
        num_tokens=seq_len_1,
    )

    dense_keys_1, dense_values_1 = fill_request_kv(
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table_1,
        seq_len=seq_len_1,
    )

    q_batch = torch.stack(
        [
            q,
            torch.randn(
                num_query_heads,
                layout.head_dim,
                device=device,
                dtype=layout.torch_dtype,
            ),
        ],
        dim=0,
    )

    batch_output = paged_attention_decode_batch_reference(
        q=q_batch,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=[block_table, block_table_1],
        seq_lens=[seq_len, seq_len_1],
    )

    dense_output_1 = dense_attention_reference(
        q=q_batch[1],
        keys=dense_keys_1,
        values=dense_values_1,
    )

    batch_diff_0 = (batch_output[0].float() - dense_output.float()).abs().max().item()
    batch_diff_1 = (batch_output[1].float() - dense_output_1.float()).abs().max().item()

    print()
    print("batch")
    print("---")
    print("batch_output_shape:", tuple(batch_output.shape))
    print("batch_diff_0:", batch_diff_0)
    print("batch_diff_1:", batch_diff_1)

    assert batch_diff_0 < 1e-2
    assert batch_diff_1 < 1e-2


if __name__ == "__main__":
    main()