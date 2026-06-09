import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout

def main() -> None:
    layout = KVCacheLayout(
        num_layers=24,
        total_blocks=8,
        block_size_tokens=16,
        num_kv_heads=2,
        head_dim=64,
        dtype="float16",
        device="cuda",
    )

    manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    request_id = "req-0"
    block_table = manager.allocate_for_tokens(
        request_id=request_id,
        num_tokens=37,
    )

    print("layout:")
    print(layout.snapshot())

    print()
    print("after allocate 37 tokens:")
    print(manager.snapshot())

    print()
    print("block_table:", block_table)

    for token_position in [0, 15, 16, 31, 32, 36]:
        physical_block_id, block_offset = layout.locate_token(
            block_table=block_table,
            token_position=token_position,
        )
        print(
            f"token_position={token_position} -> "
            f"physical_block_id={physical_block_id}, "
            f"block_offset={block_offset}"
        )

    manager.ensure_capacity_for_token(
        request_id=request_id,
        token_position=48,
    )

    print()
    print("after ensuring capacity for token 48:")
    print(manager.snapshot())

    freed = manager.free(request_id)

    print()
    print("freed:", freed)
    print("after free:")
    print(manager.snapshot())


if __name__ == "__main__":
    main()