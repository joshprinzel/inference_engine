import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from paged_attention_reference import paged_attention_decode_reference

import paged_attention_cuda


NUM_LAYERS = 2
TOTAL_BLOCKS = 32768
BLOCK_SIZE_TOKENS = 8
DTYPE = "float16"
DEVICE = "cuda"

DEFAULT_MODE = "gqa"
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 512
DEFAULT_NUM_QUERY_HEADS = 16
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_HEAD_DIM = 128
DEFAULT_WARMUP_ITERS = 50
DEFAULT_PROFILE_ITERS = 500
MAX_ALLOWED_DIFF = 1e-2


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


def make_padded_block_tables(
    block_tables: list[list[int]],
    device: str,
    pad_value: int = -1,
) -> torch.Tensor:
    max_blocks = max(len(block_table) for block_table in block_tables)

    padded = []

    for block_table in block_tables:
        row = block_table + [pad_value] * (max_blocks - len(block_table))
        padded.append(row)

    return torch.tensor(
        padded,
        dtype=torch.int32,
        device=device,
    ).contiguous()


def make_case(
    mode: str,
    batch_size: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
) -> dict[str, Any]:
    if num_query_heads < num_kv_heads:
        raise ValueError("num_query_heads must be >= num_kv_heads")

    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    torch.manual_seed(seed)

    layout = KVCacheLayout(
        num_layers=NUM_LAYERS,
        total_blocks=TOTAL_BLOCKS,
        block_size_tokens=BLOCK_SIZE_TOKENS,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=DTYPE,
        device=DEVICE,
    )

    block_manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    layer_id = 0
    seq_lens = [seq_len] * batch_size
    block_tables: list[list[int]] = []

    for request_index in range(batch_size):
        request_id = f"profile-{mode}-req-{request_index}"

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

        block_tables.append(block_table)

    q = torch.randn(
        batch_size,
        num_query_heads,
        head_dim,
        device=DEVICE,
        dtype=layout.torch_dtype,
    ).contiguous()

    block_tables_tensor = make_padded_block_tables(
        block_tables=block_tables,
        device=DEVICE,
    )

    seq_lens_tensor = torch.tensor(
        seq_lens,
        dtype=torch.int32,
        device=DEVICE,
    ).contiguous()

    return {
        "mode": mode,
        "layout": layout,
        "cache_pool": cache_pool,
        "layer_id": layer_id,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seq_lens": seq_lens,
        "seq_lens_tensor": seq_lens_tensor,
        "block_tables": block_tables,
        "block_tables_tensor": block_tables_tensor,
        "q": q,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def reference_batch_decode(
    q: torch.Tensor,
    cache_pool: KVCachePool,
    layer_id: int,
    block_tables: list[list[int]],
    seq_lens: list[int],
) -> torch.Tensor:
    outputs = []

    for batch_index, seq_len in enumerate(seq_lens):
        output_i = paged_attention_decode_reference(
            q=q[batch_index],
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_table=block_tables[batch_index],
            seq_len=seq_len,
        )

        outputs.append(output_i)

    return torch.stack(outputs, dim=0)


def run_cuda(case: dict[str, Any]) -> torch.Tensor:
    return paged_attention_cuda.paged_attention_decode_batch(
        case["q"],
        case["cache_pool"].key_cache,
        case["cache_pool"].value_cache,
        case["block_tables_tensor"],
        case["seq_lens_tensor"],
        case["layer_id"],
    )


def run_reference(case: dict[str, Any]) -> torch.Tensor:
    return reference_batch_decode(
        q=case["q"],
        cache_pool=case["cache_pool"],
        layer_id=case["layer_id"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
    )


def check_correctness(case: dict[str, Any]) -> None:
    reference_output = run_reference(case)
    cuda_output = run_cuda(case)

    diff = cuda_output.float() - reference_output.float()
    max_abs_diff = diff.abs().max().item()

    reference_finite = torch.isfinite(reference_output).all().item()
    cuda_finite = torch.isfinite(cuda_output).all().item()
    diff_finite = torch.isfinite(diff).all().item()

    passed = (
        reference_finite
        and cuda_finite
        and diff_finite
        and max_abs_diff < MAX_ALLOWED_DIFF
    )

    print("correctness:")
    print(f"  reference_finite: {reference_finite}")
    print(f"  cuda_finite:      {cuda_finite}")
    print(f"  diff_finite:      {diff_finite}")
    print(f"  max_abs_diff:     {max_abs_diff:.8f}")
    print(f"  passed:           {passed}")

    if not passed:
        raise AssertionError("CUDA output failed correctness check")


def profile_loop(
    case: dict[str, Any],
    warmup_iters: int,
    profile_iters: int,
) -> None:
    print("warmup...")
    torch.cuda.nvtx.range_push("warmup")
    for _ in range(warmup_iters):
        run_cuda(case)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("profile loop...")
    torch.cuda.nvtx.range_push("profiled_paged_attention_v7")
    for _ in range(profile_iters):
        run_cuda(case)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CUDA paged attention v7 GQA/MQA kernel."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["mha", "gqa", "mqa"],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
    )

    parser.add_argument(
        "--num-query-heads",
        type=int,
        default=DEFAULT_NUM_QUERY_HEADS,
    )

    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--head-dim",
        type=int,
        default=DEFAULT_HEAD_DIM,
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
    )

    parser.add_argument(
        "--profile-iters",
        type=int,
        default=DEFAULT_PROFILE_ITERS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--skip-correctness",
        action="store_true",
    )

    return parser.parse_args()


def resolve_num_kv_heads(
    mode: str,
    num_query_heads: int,
    explicit_num_kv_heads: int | None,
) -> int:
    if explicit_num_kv_heads is not None:
        return explicit_num_kv_heads

    if mode == "mha":
        return num_query_heads

    if mode == "gqa":
        return 4

    if mode == "mqa":
        return 1

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")

    args = parse_args()

    num_kv_heads = resolve_num_kv_heads(
        mode=args.mode,
        num_query_heads=args.num_query_heads,
        explicit_num_kv_heads=args.num_kv_heads,
    )

    case = make_case(
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=args.head_dim,
        seed=args.seed,
    )

    print("profile target:")
    print(f"  mode:              {args.mode}")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  seq_len:           {args.seq_len}")
    print(f"  num_query_heads:   {args.num_query_heads}")
    print(f"  num_kv_heads:      {num_kv_heads}")
    print(f"  q/kv:              {args.num_query_heads // num_kv_heads}")
    print(f"  head_dim:          {args.head_dim}")
    print(f"  warmup_iters:      {args.warmup_iters}")
    print(f"  profile_iters:     {args.profile_iters}")
    print(f"  extension:         {paged_attention_cuda.__file__}")
    print(f"  device:            {torch.cuda.get_device_name(0)}")

    if not args.skip_correctness:
        check_correctness(case)

    profile_loop(
        case=case,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
    )

    print("done")


if __name__ == "__main__":
    main()