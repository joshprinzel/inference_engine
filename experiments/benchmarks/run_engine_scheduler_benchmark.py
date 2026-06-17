from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.engine_scheduler import EngineScheduler
from engines.hf_decode_engine import HFDecodeEngine
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from engines.model_runner import ModelRunner
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState
from engines.synthetic_cuda_decode_engine import SyntheticCudaDecodeEngine


def run_scheduler_until_finished(
    scheduler: EngineScheduler,
    expected_finished: int,
    max_steps: int,
) -> None:
    for _ in range(max_steps):
        scheduler.step()

        if len(scheduler.finished) == expected_finished:
            return

    raise RuntimeError(
        f"Scheduler did not finish expected requests. "
        f"finished={len(scheduler.finished)}, expected={expected_finished}, "
        f"max_steps={max_steps}"
    )


def collect_backend_ms_from_history_is_unavailable() -> None:
    """
    Placeholder note.

    EngineScheduler currently only stores last_backend_ms, not a full per-step
    backend timing history. For this benchmark, synthetic CUDA backend timing
    summary is approximate unless we add a backend_ms_history field.

    The correct next cleanup is to add backend_ms_history to EngineScheduler.
    """
    return None


def run_hf_case(
    max_new_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    request_queue = RequestQueue()
    metrics_store = MetricsStore()

    kv_block_manager = KVBlockManager(
        total_blocks=256,
        block_size_tokens=16,
    )

    runner = ModelRunner()
    decode_engine = HFDecodeEngine(runner)

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
    )

    request_queue.put(
        RequestState(
            prompt="Write one short sentence about GPUs.",
            max_new_tokens=max_new_tokens,
            request_id="hf-benchmark-0",
        )
    )

    t0 = time.perf_counter()
    run_scheduler_until_finished(
        scheduler=scheduler,
        expected_finished=1,
        max_steps=max_new_tokens + 8,
    )
    t1 = time.perf_counter()

    wall_seconds = t1 - t0
    tokens_generated = scheduler.tokens_generated
    tokens_per_second = tokens_generated / wall_seconds if wall_seconds > 0 else 0.0

    result = {
        "engine": "hf",
        "num_requests": 1,
        "max_slots": 1,
        "prompt_tokens": scheduler.finished[0].prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "total_wall_seconds": wall_seconds,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "decode_iterations": scheduler.decode_steps,
        "decode_batches_built": scheduler.decode_batches_built,
        "backend_ms_last": scheduler.last_backend_ms,
        "backend_ms_median": "",
        "backend_ms_min": "",
        "backend_ms_max": "",
        "kv_used_blocks": kv_block_manager.used_block_count(),
        "kv_free_blocks": kv_block_manager.free_block_count(),
        "all_finished": all(
            request_state.status == "finished"
            for request_state in scheduler.finished
        ),
    }

    return result


def run_synthetic_cuda_case(
    batch_size: int,
    prompt_tokens: int,
    max_new_tokens: int,
    block_size_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
    attention_backend: str,
) -> dict[str, Any]:
    total_tokens_per_request = prompt_tokens + max_new_tokens
    blocks_per_request = (
        total_tokens_per_request + block_size_tokens - 1
    ) // block_size_tokens

    total_blocks = batch_size * blocks_per_request

    request_queue = RequestQueue()
    metrics_store = MetricsStore()

    kv_block_manager = KVBlockManager(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
    )

    decode_engine = SyntheticCudaDecodeEngine(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
        num_layers=1,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="float16",
        device=device,
        attention_backend=attention_backend,
    )

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=batch_size,
    )

    for request_index in range(batch_size):
        request_queue.put(
            RequestState(
                prompt=f"synthetic benchmark prompt {request_index}",
                max_new_tokens=max_new_tokens,
                request_id=f"synthetic-cuda-benchmark-{batch_size}-{prompt_tokens}-{request_index}",
            )
        )

    backend_ms_values: list[float] = []

    t0 = time.perf_counter()

    for _ in range(max_new_tokens + 8):
        scheduler.step()

        if scheduler.last_backend_ms is not None:
            backend_ms_values.append(float(scheduler.last_backend_ms))

        if len(scheduler.finished) == batch_size:
            break

    t1 = time.perf_counter()

    if len(scheduler.finished) != batch_size:
        raise RuntimeError(
            f"Synthetic CUDA case did not finish. "
            f"batch_size={batch_size}, prompt_tokens={prompt_tokens}, "
            f"finished={len(scheduler.finished)}"
        )

    failed = [
        request_state
        for request_state in scheduler.finished
        if request_state.status != "finished"
    ]

    if failed:
        details = [
            {
                "request_id": request_state.request_id,
                "status": request_state.status,
                "error": repr(request_state.error),
            }
            for request_state in failed
        ]
        raise RuntimeError(f"Synthetic CUDA case failed requests: {details}")

    wall_seconds = t1 - t0
    tokens_generated = scheduler.tokens_generated
    tokens_per_second = tokens_generated / wall_seconds if wall_seconds > 0 else 0.0

    result = {
        "engine": "synthetic-cuda",
        "num_requests": batch_size,
        "max_slots": batch_size,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "total_wall_seconds": wall_seconds,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "decode_iterations": scheduler.decode_steps,
        "decode_batches_built": scheduler.decode_batches_built,
        "backend_ms_last": scheduler.last_backend_ms,
        "backend_ms_median": statistics.median(backend_ms_values)
        if backend_ms_values
        else "",
        "backend_ms_min": min(backend_ms_values) if backend_ms_values else "",
        "backend_ms_max": max(backend_ms_values) if backend_ms_values else "",
        "kv_used_blocks": kv_block_manager.used_block_count(),
        "kv_free_blocks": kv_block_manager.free_block_count(),
        "all_finished": all(
            request_state.status == "finished"
            for request_state in scheduler.finished
        ),
    }

    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "engine",
        "num_requests",
        "max_slots",
        "prompt_tokens",
        "max_new_tokens",
        "total_wall_seconds",
        "tokens_generated",
        "tokens_per_second",
        "decode_iterations",
        "decode_batches_built",
        "backend_ms_last",
        "backend_ms_median",
        "backend_ms_min",
        "backend_ms_max",
        "kv_used_blocks",
        "kv_free_blocks",
        "all_finished",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "benchmarks" / "engine_scheduler_benchmark.csv",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip Hugging Face benchmark case.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic CUDA benchmark cases.",
    )
    parser.add_argument("--hf-max-new-tokens", type=int, default=16)
    parser.add_argument("--synthetic-max-new-tokens", type=int, default=32)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attention-backend", type=str, default="cuda")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []

    if not args.skip_hf:
        print("Running HFDecodeEngine benchmark...")
        hf_row = run_hf_case(
            max_new_tokens=args.hf_max_new_tokens,
            output_path=args.output,
        )
        rows.append(hf_row)
        print(hf_row)

    if not args.skip_synthetic:
        print("Running SyntheticCudaDecodeEngine benchmarks...")

        for prompt_tokens in [128, 512]:
            for batch_size in [1, 4, 8, 16, 32]:
                print(
                    "Synthetic CUDA case:",
                    {
                        "batch_size": batch_size,
                        "prompt_tokens": prompt_tokens,
                    },
                )

                row = run_synthetic_cuda_case(
                    batch_size=batch_size,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=args.synthetic_max_new_tokens,
                    block_size_tokens=args.block_size_tokens,
                    num_query_heads=args.num_query_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_dim=args.head_dim,
                    device=args.device,
                    attention_backend=args.attention_backend,
                )

                rows.append(row)
                print(row)

    write_csv(args.output, rows)

    print(f"Wrote benchmark CSV to {args.output}")


if __name__ == "__main__":
    main()