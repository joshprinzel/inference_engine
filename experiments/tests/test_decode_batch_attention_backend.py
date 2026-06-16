from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from attention_backend import build_attention_backend
from decode_batch import build_decode_batch
from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from request_state import RequestState


def make_request_state(
    request_id: str,
    prompt_tokens: int,
    generated_tokens: int,
    next_token_id: int,
) -> RequestState:
    request = RequestState(
        request_id=request_id,
        prompt="synthetic",
        max_new_tokens=16,
    )

    request.prompt_tokens = prompt_tokens
    request.generated_tokens = generated_tokens
    request.next_token = torch.tensor([[next_token_id]], dtype=torch.int64)
    request.mark_decoding()

    return request


def fill_cache_for_request(
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
        value = torch.randn_like(key)

        cache_pool.write_request_token(
            layer_id=layer_id,
            block_table=block_table,
            token_position=token_position,
            key=key,
            value=value,
        )


def time_backend(
    backend,
    q: torch.Tensor,
    cache_pool: KVCachePool,
    layer_id: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    iters: int,
) -> tuple[torch.Tensor, float]:
    if q.device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    out = None

    for _ in range(iters):
        if q.device.type == "cuda":
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
            times.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            out = backend.decode(
                q=q,
                cache_pool=cache_pool,
                layer_id=layer_id,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    assert out is not None
    return out, float(torch.tensor(times).median().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--total-blocks", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this integration test")

    torch.manual_seed(0)

    device = "cuda"
    dtype = "float16"
    layer_id = 0

    blocks_per_request = (args.seq_len + args.block_size - 1) // args.block_size
    required_blocks = args.batch_size * blocks_per_request

    if args.total_blocks < required_blocks:
        new_total_blocks = required_blocks
        print(
            f"Overriding total_blocks={args.total_blocks} to {new_total_blocks}. "
            f"Required: batch_size({args.batch_size}) * blocks_per_request({blocks_per_request})"
        )
        args.total_blocks = new_total_blocks

    layout = KVCacheLayout(
        num_layers=args.num_layers,
        total_blocks=args.total_blocks,
        block_size_tokens=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    block_manager = KVBlockManager(
        total_blocks=args.total_blocks,
        block_size_tokens=args.block_size,
    )

    request_states: list[RequestState] = []

    for i in range(args.batch_size):
        request_id = f"req-{i}"

        block_table = block_manager.allocate_for_tokens(
            request_id=request_id,
            num_tokens=args.seq_len,
        )

        fill_cache_for_request(
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_table=block_table,
            seq_len=args.seq_len,
        )

        # build_decode_batch computes:
        # position = prompt_tokens + generated_tokens
        # seq_len = position + 1
        #
        # So to get desired seq_len, set position = seq_len - 1.
        request_state = make_request_state(
            request_id=request_id,
            prompt_tokens=args.seq_len,
            generated_tokens=0,
            next_token_id=100 + i,
        )

        request_states.append(request_state)

    decode_batch = build_decode_batch(
        request_states=request_states,
        kv_block_manager=block_manager,
        device=device,
    )

    q = torch.randn(
        decode_batch.batch_size,
        args.num_query_heads,
        args.head_dim,
        device=device,
        dtype=torch.float16,
    )

    ref_backend = build_attention_backend("reference")
    cuda_backend = build_attention_backend("cuda")

    out_ref = ref_backend.decode(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=decode_batch.block_tables,
        seq_lens=decode_batch.seq_lens,
    )

    out_cuda = cuda_backend.decode(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=decode_batch.block_tables,
        seq_lens=decode_batch.seq_lens,
    )

    max_abs_diff = (out_ref - out_cuda).abs().max().item()
    mean_abs_diff = (out_ref - out_cuda).abs().mean().item()

    print("decode_batch:")
    print(decode_batch.snapshot())
    print(f"max_abs_diff:  {max_abs_diff:.8f}")
    print(f"mean_abs_diff: {mean_abs_diff:.8f}")

    if max_abs_diff > 1e-2:
        raise AssertionError(f"max_abs_diff too high: {max_abs_diff}")

    _, ref_med_ms = time_backend(
        backend=ref_backend,
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=decode_batch.block_tables,
        seq_lens=decode_batch.seq_lens,
        iters=max(3, args.iters // 10),
    )

    _, cuda_med_ms = time_backend(
        backend=cuda_backend,
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=decode_batch.block_tables,
        seq_lens=decode_batch.seq_lens,
        iters=args.iters,
    )

    print(f"ref_med_ms:  {ref_med_ms:.6f}")
    print(f"cuda_med_ms: {cuda_med_ms:.6f}")
    print(f"speedup:     {ref_med_ms / cuda_med_ms:.2f}x")
    print("passed")


if __name__ == "__main__":
    main()