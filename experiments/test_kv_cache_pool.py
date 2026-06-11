import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layout = KVCacheLayout(
        num_layers=2,
        total_blocks=8,
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=64,
        dtype="float16",
        device=device,
    )

    block_manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    request_id = "req-0"
    block_table = block_manager.allocate_for_tokens(
        request_id=request_id,
        num_tokens=37,
    )

    print("cache pool snapshot:")
    print(cache_pool.snapshot())

    print()
    print("block table:")
    print(block_table)

    layer_id = 1
    token_position = 32

    key = torch.randn(
        layout.num_kv_heads,
        layout.head_dim,
        device=device,
        dtype=layout.torch_dtype,
    )

    value = torch.randn(
        layout.num_kv_heads,
        layout.head_dim,
        device=device,
        dtype=layout.torch_dtype,
    )

    physical_block_id, block_offset = cache_pool.write_request_token(
        layer_id=layer_id,
        block_table=block_table,
        token_position=token_position,
        key=key,
        value=value,
    )

    read_key, read_value = cache_pool.read_request_token(
        layer_id=layer_id,
        block_table=block_table,
        token_position=token_position,
    )

    key_matches = torch.equal(key, read_key)
    value_matches = torch.equal(value, read_value)

    print()
    print("write/read address:")
    print(
        {
            "layer_id": layer_id,
            "token_position": token_position,
            "physical_block_id": physical_block_id,
            "block_offset": block_offset,
        }
    )

    print()
    print("validation:")
    print(
        {
            "key_matches": key_matches,
            "value_matches": value_matches,
        }
    )

    assert key_matches
    assert value_matches


if __name__ == "__main__":
    main()