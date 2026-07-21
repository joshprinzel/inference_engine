from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState


@dataclass
class RequestTrace:
    request_id: str
    prompt_tokens: int | None = None
    arrival_step: int | None = None
    admitted_step: int | None = None
    first_decode_step: int | None = None
    finished_step: int | None = None
    generated_tokens: int = 0
    decode_steps: list[int] | None = None
    decode_event_wall_times_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.decode_steps is None:
            self.decode_steps = []
        if self.decode_event_wall_times_ms is None:
            self.decode_event_wall_times_ms = []


def make_prompt(num_words: int, prefix: str) -> str:
    words = [prefix]
    words.extend(f"token{i}" for i in range(num_words))
    return " ".join(words)


def get_slots(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    slots = snapshot.get("slots", [])
    return [slot for slot in slots if slot is not None]


def get_slot_by_request_id(
    snapshot: dict[str, Any],
    request_id: str,
) -> dict[str, Any] | None:
    for slot in get_slots(snapshot):
        if str(slot.get("request_id")) == request_id:
            return slot
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(round((p / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]


def percent_reduction(
    *,
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        return 0.0

    return ((baseline - candidate) / baseline) * 100.0


def format_change_sentence(
    *,
    metric_name: str,
    full_value: float,
    chunked_value: float,
) -> str:
    reduction = percent_reduction(
        baseline=full_value,
        candidate=chunked_value,
    )

    if reduction >= 0:
        return (
            f"chunked prefill reduced {metric_name} "
            f"from {full_value:.3f} ms to {chunked_value:.3f} ms "
            f"({reduction:.1f}% reduction)."
        )

    return (
        f"chunked prefill increased {metric_name} "
        f"from {full_value:.3f} ms to {chunked_value:.3f} ms "
        f"({abs(reduction):.1f}% increase)."
    )


def summarize_tbt(
    *,
    decode_steps: list[int],
    decode_event_wall_times_ms: list[float],
) -> dict[str, float]:
    """
    TBT = time-between-tokens for an already-decoding request.

    This benchmark measures the Sarathi-style interference case:
    a short request is already decoding, then a long prefill request enters
    the system. Full prefill can create one large TBT stall; chunked prefill
    should bound that stall by splitting prompt prefill into smaller chunks.
    """

    step_gaps: list[float] = []
    wall_gaps_ms: list[float] = []

    for prev_step, next_step in zip(decode_steps, decode_steps[1:]):
        step_gaps.append(float(next_step - prev_step))

    for prev_ms, next_ms in zip(
        decode_event_wall_times_ms,
        decode_event_wall_times_ms[1:],
    ):
        wall_gaps_ms.append(next_ms - prev_ms)

    return {
        "avg_tbt_steps": statistics.mean(step_gaps) if step_gaps else 0.0,
        "p95_tbt_steps": percentile(step_gaps, 95),
        "max_tbt_steps": max(step_gaps) if step_gaps else 0.0,
        "avg_tbt_ms": statistics.mean(wall_gaps_ms) if wall_gaps_ms else 0.0,
        "p50_tbt_ms": statistics.median(wall_gaps_ms) if wall_gaps_ms else 0.0,
        "p95_tbt_ms": percentile(wall_gaps_ms, 95),
        "max_tbt_ms": max(wall_gaps_ms) if wall_gaps_ms else 0.0,
        "num_tbt_samples": float(len(wall_gaps_ms)),
    }


def run_benchmark(
    *,
    mode: str,
    prefill_budget: int | None,
    short_prompt_words: int,
    long_prompt_words: int,
    short_max_new_tokens: int,
    long_max_new_tokens: int,
    max_slots: int,
    total_kv_blocks: int,
    block_size_tokens: int,
    long_request_arrival_decode_tokens: int,
    max_steps: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    decode_engine = CustomLlamaDecodeEngine(
        attention_backend_name="cuda",
        total_kv_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
    )
    kv_block_manager = KVBlockManager(
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
    )
    request_queue = RequestQueue()
    metrics_store = MetricsStore()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=max_slots,
        max_scheduled_tokens_per_step=prefill_budget,
    )

    short_request = RequestState(
        request_id="short-1",
        prompt=make_prompt(short_prompt_words, "Short request."),
        max_new_tokens=short_max_new_tokens,
    )
    long_request = RequestState(
        request_id="long-1",
        prompt=make_prompt(long_prompt_words, "Long request."),
        max_new_tokens=long_max_new_tokens,
    )

    traces = {
        "short-1": RequestTrace(request_id="short-1"),
        "long-1": RequestTrace(request_id="long-1"),
    }

    current_step = 0
    long_submitted = False
    previous_generated_tokens = {
        "short-1": 0,
        "long-1": 0,
    }

    request_queue.put(short_request)
    traces["short-1"].arrival_step = 0

    run_start_ns = time.perf_counter_ns()

    for current_step in range(1, max_steps + 1):
        scheduler.step()
        after_step_ns = time.perf_counter_ns()
        elapsed_wall_ms = (after_step_ns - run_start_ns) / 1_000_000.0

        snapshot = scheduler.snapshot()

        for request in [short_request, long_request]:
            request_id = str(request.request_id)
            trace = traces[request_id]
            slot = get_slot_by_request_id(snapshot, request_id)

            if trace.prompt_tokens is None and request.prompt_tokens > 0:
                trace.prompt_tokens = request.prompt_tokens

            if trace.admitted_step is None and slot is not None:
                trace.admitted_step = current_step

            generated_delta = (
                request.generated_tokens - previous_generated_tokens[request_id]
            )

            if generated_delta > 0:
                if trace.first_decode_step is None:
                    trace.first_decode_step = current_step

                trace.decode_steps.append(current_step)
                trace.decode_event_wall_times_ms.append(elapsed_wall_ms)
                trace.generated_tokens = request.generated_tokens

            previous_generated_tokens[request_id] = request.generated_tokens

            if trace.finished_step is None and request.status == "finished":
                trace.finished_step = current_step

        if (
            not long_submitted
            and short_request.generated_tokens >= long_request_arrival_decode_tokens
        ):
            request_queue.put(long_request)
            long_submitted = True
            traces["long-1"].arrival_step = current_step

        if short_request.status == "finished" and long_request.status == "finished":
            break

    run_end_ns = time.perf_counter_ns()
    total_wall_time_ms = (run_end_ns - run_start_ns) / 1_000_000.0

    short_trace = traces["short-1"]
    long_trace = traces["long-1"]

    tbt_summary = summarize_tbt(
        decode_steps=short_trace.decode_steps or [],
        decode_event_wall_times_ms=short_trace.decode_event_wall_times_ms or [],
    )

    result = {
        "mode": mode,
        "prefill_budget": "unlimited" if prefill_budget is None else prefill_budget,
        "max_slots": max_slots,
        "total_kv_blocks": total_kv_blocks,
        "block_size_tokens": block_size_tokens,
        "short_prompt_words": short_prompt_words,
        "long_prompt_words": long_prompt_words,
        "short_prompt_tokens": short_trace.prompt_tokens,
        "long_prompt_tokens": long_trace.prompt_tokens,
        "short_generated_tokens": short_request.generated_tokens,
        "long_generated_tokens": long_request.generated_tokens,
        "short_arrival_step": short_trace.arrival_step,
        "long_arrival_step": long_trace.arrival_step,
        "long_arrival_after_short_decode_tokens": long_request_arrival_decode_tokens,
        "short_first_decode_step": short_trace.first_decode_step,
        "long_first_decode_step": long_trace.first_decode_step,
        "short_finished_step": short_trace.finished_step,
        "long_finished_step": long_trace.finished_step,
        "num_tbt_samples": int(tbt_summary["num_tbt_samples"]),
        "avg_short_tbt_steps": tbt_summary["avg_tbt_steps"],
        "p95_short_tbt_steps": tbt_summary["p95_tbt_steps"],
        "max_short_tbt_steps": tbt_summary["max_tbt_steps"],
        "avg_short_tbt_ms": tbt_summary["avg_tbt_ms"],
        "p50_short_tbt_ms": tbt_summary["p50_tbt_ms"],
        "p95_short_tbt_ms": tbt_summary["p95_tbt_ms"],
        "max_short_tbt_ms": tbt_summary["max_tbt_ms"],
        "total_steps": current_step,
        "total_wall_time_ms": total_wall_time_ms,
    }

    return result


def write_results(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nSarathi-Style Chunked Prefill TBT Benchmark")
    print("=" * 76)
    print(
        "Workload: one short request enters decode, then one long-prompt request "
        "is admitted. TBT measures time-between-tokens for the already-decoding "
        "short request."
    )
    print()

    for row in rows:
        print(
            f"{row['mode']:>8} | "
            f"budget={row['prefill_budget']} | "
            f"long_prompt_tokens={row['long_prompt_tokens']} | "
            f"short_generated_tokens={row['short_generated_tokens']} | "
            f"samples={row['num_tbt_samples']} | "
            f"avg_tbt_ms={row['avg_short_tbt_ms']:.3f} | "
            f"p50_tbt_ms={row['p50_short_tbt_ms']:.3f} | "
            f"p95_tbt_ms={row['p95_short_tbt_ms']:.3f} | "
            f"max_tbt_ms={row['max_short_tbt_ms']:.3f} | "
            f"total_wall_ms={row['total_wall_time_ms']:.3f}"
        )

    full_row = next((row for row in rows if row["mode"] == "full"), None)
    chunked_row = next((row for row in rows if row["mode"] == "chunked"), None)

    if full_row is None or chunked_row is None:
        return

    full_avg = float(full_row["avg_short_tbt_ms"])
    chunked_avg = float(chunked_row["avg_short_tbt_ms"])

    full_p95 = float(full_row["p95_short_tbt_ms"])
    chunked_p95 = float(chunked_row["p95_short_tbt_ms"])

    full_max = float(full_row["max_short_tbt_ms"])
    chunked_max = float(chunked_row["max_short_tbt_ms"])

    print("\nHeadline candidates:")

    if full_avg > 0:
        print(
            format_change_sentence(
                metric_name="average short-request TBT",
                full_value=full_avg,
                chunked_value=chunked_avg,
            )
        )

    if full_p95 > 0:
        print(
            format_change_sentence(
                metric_name="p95 short-request TBT",
                full_value=full_p95,
                chunked_value=chunked_p95,
            )
        )

    if full_max > 0:
        print(
            format_change_sentence(
                metric_name="worst-case short-request TBT",
                full_value=full_max,
                chunked_value=chunked_max,
            )
        )

    print("\nRecommended resume-safe wording:")
    max_reduction = percent_reduction(
        baseline=full_max,
        candidate=chunked_max,
    )
    if max_reduction > 0:
        print(
            f"Reduced worst-case short-request time-between-tokens from "
            f"{full_max / 1000.0:.2f}s to {chunked_max:.1f}ms "
            f"({max_reduction:.1f}% reduction) under long-prompt prefill pressure "
            f"using token-budgeted chunked prefill."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunked-budget", type=int, default=16)
    parser.add_argument("--short-prompt-words", type=int, default=8)
    parser.add_argument("--long-prompt-words", type=int, default=512)
    parser.add_argument("--short-max-new-tokens", type=int, default=48)
    parser.add_argument("--long-max-new-tokens", type=int, default=4)
    parser.add_argument("--max-slots", type=int, default=2)
    parser.add_argument("--total-kv-blocks", type=int, default=512)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--long-arrival-decode-tokens", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/chunked_prefill_tbt.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_result = run_benchmark(
        mode="full",
        prefill_budget=None,
        short_prompt_words=args.short_prompt_words,
        long_prompt_words=args.long_prompt_words,
        short_max_new_tokens=args.short_max_new_tokens,
        long_max_new_tokens=args.long_max_new_tokens,
        max_slots=args.max_slots,
        total_kv_blocks=args.total_kv_blocks,
        block_size_tokens=args.block_size_tokens,
        long_request_arrival_decode_tokens=args.long_arrival_decode_tokens,
        max_steps=args.max_steps,
    )

    torch.cuda.empty_cache()

    chunked_result = run_benchmark(
        mode="chunked",
        prefill_budget=args.chunked_budget,
        short_prompt_words=args.short_prompt_words,
        long_prompt_words=args.long_prompt_words,
        short_max_new_tokens=args.short_max_new_tokens,
        long_max_new_tokens=args.long_max_new_tokens,
        max_slots=args.max_slots,
        total_kv_blocks=args.total_kv_blocks,
        block_size_tokens=args.block_size_tokens,
        long_request_arrival_decode_tokens=args.long_arrival_decode_tokens,
        max_steps=args.max_steps,
    )

    rows = [full_result, chunked_result]
    write_results(output_path=args.output, rows=rows)
    print_summary(rows)

    print(f"\nWrote results to: {args.output}")


if __name__ == "__main__":
    main()