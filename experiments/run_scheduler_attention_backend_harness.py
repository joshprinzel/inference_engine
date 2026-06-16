from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention_backend import build_attention_backend
from decode_batch import build_decode_batch
from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from request_state import RequestState


def make_requests(
    batch_size: int,
    prompt_tokens: int,
    max_new_tokens: int,
) -> list[RequestState]:
    requests: list[RequestState] = []

    for i in range(batch_size):
        request = RequestState(
            prompt=f"synthetic request {i}",
            max_new_tokens=max_new_tokens,
            request_id=f"req-{i}",
        )
        request.prompt_tokens = prompt_tokens
        request.generated_tokens = 0
        request.next_token = torch.tensor([[100 + i]], dtype=torch.int64)
        request.mark_admitted()
        request.mark_decoding()
        requests.append(request)

    return requests


def fill_prompt_kv(
    cache_pool: KVCachePool,
    layer_id: int,
    block_table: list[int],
    prompt_tokens: int,
) -> None:
    for token_position in range(prompt_tokens):
        key = torch.randn(
            cache_pool.layout.num_kv_heads,
            cache_pool.layout.head_dim,
            device=cache_pool.key_cache.device,
            dtype=cache_pool.key_cache.dtype,
        )
        value = torch.randn_like(key)

        cache_pool.write_request_token(
            layer_id=layer_id,
            block_table=block_table,
            token_position=token_position,
            key=key,
            value=value,
        )


def write_generated_token_kv(
    cache_pool: KVCachePool,
    layer_id: int,
    block_table: list[int],
    token_position: int,
) -> None:
    key = torch.randn(
        cache_pool.layout.num_kv_heads,
        cache_pool.layout.head_dim,
        device=cache_pool.key_cache.device,
        dtype=cache_pool.key_cache.dtype,
    )
    value = torch.randn_like(key)

    cache_pool.write_request_token(
        layer_id=layer_id,
        block_table=block_table,
        token_position=token_position,
        key=key,
        value=value,
    )


def cuda_time_backend_decode(
    backend,
    q: torch.Tensor,
    cache_pool: KVCachePool,
    layer_id: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = backend.decode(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )
    end.record()

    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "reference"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--total-blocks", type=int, default=0)
    parser.add_argument("--results-csv", type=str, default="results/scheduler_attention_backend_harness.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness")

    torch.manual_seed(0)

    device = "cuda"
    dtype = "float16"
    layer_id = 0

    # Need enough blocks for prompt + generated tokens.
    max_tokens_per_request = args.prompt_tokens + args.max_new_tokens
    blocks_per_request = (max_tokens_per_request + args.block_size - 1) // args.block_size
    required_blocks = args.batch_size * blocks_per_request

    total_blocks = args.total_blocks
    if total_blocks <= 0:
        total_blocks = required_blocks

    if total_blocks < required_blocks:
        raise ValueError(
            f"total_blocks={total_blocks} is too small. "
            f"Need at least {required_blocks} for batch_size={args.batch_size}, "
            f"prompt_tokens={args.prompt_tokens}, max_new_tokens={args.max_new_tokens}, "
            f"block_size={args.block_size}."
        )

    layout = KVCacheLayout(
        num_layers=args.num_layers,
        total_blocks=total_blocks,
        block_size_tokens=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    block_manager = KVBlockManager(
        total_blocks=total_blocks,
        block_size_tokens=args.block_size,
    )

    request_states = make_requests(
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    # Allocate initial prompt blocks and fill prompt KV.
    for request_state in request_states:
        block_table = block_manager.allocate_for_tokens(
            request_id=str(request_state.request_id),
            num_tokens=args.prompt_tokens,
        )
        request_state.block_table = block_table

        fill_prompt_kv(
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_table=block_table,
            prompt_tokens=args.prompt_tokens,
        )

    backend = build_attention_backend(args.backend)

    results_path = REPO_ROOT / args.results_csv
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    wall_start = time.perf_counter()
    decode_step = 0
    total_backend_ms = 0.0
    total_tokens_emitted = 0

    while True:
        active = [r for r in request_states if r.status != "finished"]
        if not active:
            break

        decode_batch = build_decode_batch(
            request_states=active,
            kv_block_manager=block_manager,
            device=device
        )

        if decode_batch.batch_size == 0:
            break

        q = torch.randn(
            decode_batch.batch_size,
            args.num_query_heads,
            args.head_dim,
            device=device,
            dtype=torch.float16,
        )

        if args.backend == "cuda":
            _, backend_ms = cuda_time_backend_decode(
                backend=backend,
                q=q,
                cache_pool=cache_pool,
                layer_id=layer_id,
                block_tables=decode_batch.block_tables,
                seq_lens=decode_batch.seq_lens,
            )
        else:
            t0 = time.perf_counter()
            _ = backend.decode(
                q=q,
                cache_pool=cache_pool,
                layer_id=layer_id,
                block_tables=decode_batch.block_tables,
                seq_lens=decode_batch.seq_lens,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            backend_ms = (t1 - t0) * 1000.0

        total_backend_ms += backend_ms

        # Fake token selection + KV append.
        # This is intentionally not full model execution. It tests runtime plumbing.
        for local_idx, request_state in enumerate(active):
            token_position = request_state.prompt_tokens + request_state.generated_tokens

            block_manager.ensure_capacity_for_token(
                request_id=str(request_state.request_id),
                token_position=token_position,
            )

            block_table = block_manager.get_block_tables(str(request_state.request_id))
            request_state.block_table = block_table

            write_generated_token_kv(
                cache_pool=cache_pool,
                layer_id=layer_id,
                block_table=block_table,
                token_position=token_position,
            )

            request_state.generated_tokens += 1
            total_tokens_emitted += 1

            # Deterministic fake next token.
            request_state.next_token = torch.tensor(
                [[1000 + request_state.generated_tokens]],
                dtype=torch.int64,
            )

            if request_state.is_finished():
                request_state.mark_finished()
                block_manager.free(str(request_state.request_id))

        rows.append(
            {
                "decode_step": decode_step,
                "active_batch_size": decode_batch.batch_size,
                "backend": args.backend,
                "backend_ms": backend_ms,
                "kv_used_blocks": block_manager.used_block_count(),
                "kv_free_blocks": block_manager.free_block_count(),
                "kv_utilization": block_manager.utilization(),
                "total_tokens_emitted": total_tokens_emitted,
            }
        )

        decode_step += 1

    wall_end = time.perf_counter()
    wall_seconds = wall_end - wall_start

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "decode_step",
                "active_batch_size",
                "backend",
                "backend_ms",
                "kv_used_blocks",
                "kv_free_blocks",
                "kv_utilization",
                "total_tokens_emitted",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    backend_times = torch.tensor([row["backend_ms"] for row in rows], dtype=torch.float32)

    print("scheduler attention backend harness")
    print("-----------------------------------")
    print(f"backend:              {args.backend}")
    print(f"batch_size:           {args.batch_size}")
    print(f"prompt_tokens:        {args.prompt_tokens}")
    print(f"max_new_tokens:       {args.max_new_tokens}")
    print(f"decode_steps:         {decode_step}")
    print(f"total_tokens_emitted: {total_tokens_emitted}")
    print(f"wall_seconds:         {wall_seconds:.6f}")
    print(f"tokens_per_second:    {total_tokens_emitted / wall_seconds:.2f}")
    print(f"backend_med_ms:       {backend_times.median().item():.6f}")
    print(f"backend_min_ms:       {backend_times.min().item():.6f}")
    print(f"backend_max_ms:       {backend_times.max().item():.6f}")
    print(f"results_csv:          {results_path}")
    print("final_block_manager:")
    print(block_manager.snapshot())


if __name__ == "__main__":
    main()