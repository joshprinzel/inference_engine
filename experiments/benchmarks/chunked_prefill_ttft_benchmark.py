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
    kind: str
    arrival_step: int | None = None
    arrival_wall_ms: float | None = None
    admitted_step: int | None = None
    first_decode_step: int | None = None
    first_decode_wall_ms: float | None = None
    finished_step: int | None = None
    finished_wall_ms: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int = 0

    @property
    def ttft_steps(self) -> int | None:
        if self.arrival_step is None or self.first_decode_step is None:
            return None
        return self.first_decode_step - self.arrival_step

    @property
    def ttft_wall_ms(self) -> float | None:
        if self.arrival_wall_ms is None or self.first_decode_wall_ms is None:
            return None
        return self.first_decode_wall_ms - self.arrival_wall_ms

    @property
    def completion_wall_ms(self) -> float | None:
        if self.arrival_wall_ms is None or self.finished_wall_ms is None:
            return None
        return self.finished_wall_ms - self.arrival_wall_ms


def make_prompt(num_words: int, prefix: str) -> str:
    words = [prefix]
    words.extend(f"token{i}" for i in range(num_words))
    return " ".join(words)


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


def submit_request(
    *,
    request_queue: RequestQueue,
    request: RequestState,
    trace: RequestTrace,
    current_step: int,
    elapsed_wall_ms: float,
) -> None:
    request_queue.put(request)
    trace.arrival_step = current_step
    trace.arrival_wall_ms = elapsed_wall_ms


def summarize_short_requests(
    *,
    traces: dict[str, RequestTrace],
) -> dict[str, float]:
    short_traces = [
        trace
        for trace in traces.values()
        if trace.kind == "short"
    ]

    ttft_values = [
        trace.ttft_wall_ms
        for trace in short_traces
        if trace.ttft_wall_ms is not None
    ]
    completion_values = [
        trace.completion_wall_ms
        for trace in short_traces
        if trace.completion_wall_ms is not None
    ]
    ttft_step_values = [
        float(trace.ttft_steps)
        for trace in short_traces
        if trace.ttft_steps is not None
    ]

    return {
        "num_short_requests": float(len(short_traces)),
        "num_short_requests_with_first_decode": float(len(ttft_values)),
        "avg_short_ttft_ms": statistics.mean(ttft_values) if ttft_values else 0.0,
        "p50_short_ttft_ms": statistics.median(ttft_values) if ttft_values else 0.0,
        "p95_short_ttft_ms": percentile(ttft_values, 95),
        "max_short_ttft_ms": max(ttft_values) if ttft_values else 0.0,
        "avg_short_ttft_steps": (
            statistics.mean(ttft_step_values)
            if ttft_step_values
            else 0.0
        ),
        "avg_short_completion_ms": (
            statistics.mean(completion_values)
            if completion_values
            else 0.0
        ),
        "max_short_completion_ms": (
            max(completion_values)
            if completion_values
            else 0.0
        ),
    }


def run_benchmark(
    *,
    mode: str,
    prefill_budget: int | None,
    long_prompt_words: int,
    short_prompt_words: int,
    num_short_requests: int,
    short_arrival_start_step: int,
    short_arrival_interval_steps: int,
    long_max_new_tokens: int,
    short_max_new_tokens: int,
    max_slots: int,
    total_kv_blocks: int,
    block_size_tokens: int,
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

    long_request = RequestState(
        request_id="long-1",
        prompt=make_prompt(long_prompt_words, "Long request."),
        max_new_tokens=long_max_new_tokens,
    )

    short_requests = [
        RequestState(
            request_id=f"short-{index}",
            prompt=make_prompt(short_prompt_words, f"Short request {index}."),
            max_new_tokens=short_max_new_tokens,
        )
        for index in range(1, num_short_requests + 1)
    ]

    all_requests = [long_request, *short_requests]

    traces: dict[str, RequestTrace] = {
        "long-1": RequestTrace(
            request_id="long-1",
            kind="long",
        )
    }

    for request in short_requests:
        traces[str(request.request_id)] = RequestTrace(
            request_id=str(request.request_id),
            kind="short",
        )

    previous_generated_tokens = {
        str(request.request_id): 0
        for request in all_requests
    }

    submitted_request_ids: set[str] = set()

    run_start_ns = time.perf_counter_ns()

    # Submit the long request before the first scheduler step.
    submit_request(
        request_queue=request_queue,
        request=long_request,
        trace=traces["long-1"],
        current_step=0,
        elapsed_wall_ms=0.0,
    )
    submitted_request_ids.add("long-1")

    current_step = 0

    for current_step in range(1, max_steps + 1):
        before_step_elapsed_ms = (
            (time.perf_counter_ns() - run_start_ns) / 1_000_000.0
        )

        for index, short_request in enumerate(short_requests, start=1):
            request_id = str(short_request.request_id)
            arrival_step = (
                short_arrival_start_step
                + ((index - 1) * short_arrival_interval_steps)
            )

            if (
                current_step >= arrival_step
                and request_id not in submitted_request_ids
            ):
                submit_request(
                    request_queue=request_queue,
                    request=short_request,
                    trace=traces[request_id],
                    current_step=current_step,
                    elapsed_wall_ms=before_step_elapsed_ms,
                )
                submitted_request_ids.add(request_id)

        scheduler.step()
        after_step_ns = time.perf_counter_ns()
        elapsed_wall_ms = (after_step_ns - run_start_ns) / 1_000_000.0

        snapshot = scheduler.snapshot()

        for request in all_requests:
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
                    trace.first_decode_wall_ms = elapsed_wall_ms

                trace.generated_tokens = request.generated_tokens

            previous_generated_tokens[request_id] = request.generated_tokens

            if trace.finished_step is None and request.status == "finished":
                trace.finished_step = current_step
                trace.finished_wall_ms = elapsed_wall_ms

        if all(request.status == "finished" for request in all_requests):
            break

    run_end_ns = time.perf_counter_ns()
    total_wall_time_ms = (run_end_ns - run_start_ns) / 1_000_000.0

    short_summary = summarize_short_requests(traces=traces)
    long_trace = traces["long-1"]

    result = {
        "mode": mode,
        "prefill_budget": "unlimited" if prefill_budget is None else prefill_budget,
        "max_slots": max_slots,
        "total_kv_blocks": total_kv_blocks,
        "block_size_tokens": block_size_tokens,
        "long_prompt_words": long_prompt_words,
        "short_prompt_words": short_prompt_words,
        "long_prompt_tokens": long_trace.prompt_tokens,
        "num_short_requests": int(short_summary["num_short_requests"]),
        "num_short_requests_with_first_decode": int(
            short_summary["num_short_requests_with_first_decode"]
        ),
        "short_arrival_start_step": short_arrival_start_step,
        "short_arrival_interval_steps": short_arrival_interval_steps,
        "avg_short_ttft_ms": short_summary["avg_short_ttft_ms"],
        "p50_short_ttft_ms": short_summary["p50_short_ttft_ms"],
        "p95_short_ttft_ms": short_summary["p95_short_ttft_ms"],
        "max_short_ttft_ms": short_summary["max_short_ttft_ms"],
        "avg_short_ttft_steps": short_summary["avg_short_ttft_steps"],
        "avg_short_completion_ms": short_summary["avg_short_completion_ms"],
        "max_short_completion_ms": short_summary["max_short_completion_ms"],
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
    print("\nChunked Prefill TTFT Benchmark")
    print("=" * 72)

    for row in rows:
        print(
            f"{row['mode']:>8} | "
            f"budget={row['prefill_budget']} | "
            f"long_prompt_tokens={row['long_prompt_tokens']} | "
            f"short_requests={row['num_short_requests_with_first_decode']}/"
            f"{row['num_short_requests']} | "
            f"avg_ttft_ms={row['avg_short_ttft_ms']:.3f} | "
            f"p50_ttft_ms={row['p50_short_ttft_ms']:.3f} | "
            f"p95_ttft_ms={row['p95_short_ttft_ms']:.3f} | "
            f"max_ttft_ms={row['max_short_ttft_ms']:.3f} | "
            f"avg_completion_ms={row['avg_short_completion_ms']:.3f} | "
            f"total_wall_ms={row['total_wall_time_ms']:.3f}"
        )

    full_row = next((row for row in rows if row["mode"] == "full"), None)
    chunked_row = next((row for row in rows if row["mode"] == "chunked"), None)

    if full_row is None or chunked_row is None:
        return

    full_avg_ttft = float(full_row["avg_short_ttft_ms"])
    chunked_avg_ttft = float(chunked_row["avg_short_ttft_ms"])

    full_p95_ttft = float(full_row["p95_short_ttft_ms"])
    chunked_p95_ttft = float(chunked_row["p95_short_ttft_ms"])

    full_max_ttft = float(full_row["max_short_ttft_ms"])
    chunked_max_ttft = float(chunked_row["max_short_ttft_ms"])

    print("\nHeadline candidates:")

    if full_avg_ttft > 0:
        avg_reduction = percent_reduction(
            baseline=full_avg_ttft,
            candidate=chunked_avg_ttft,
        )
        if avg_reduction >= 0:
            print(
                f"chunked prefill reduced average short-request TTFT "
                f"from {full_avg_ttft:.3f} ms to {chunked_avg_ttft:.3f} ms "
                f"({avg_reduction:.1f}% reduction)."
            )
        else:
            print(
                f"chunked prefill increased average short-request TTFT "
                f"from {full_avg_ttft:.3f} ms to {chunked_avg_ttft:.3f} ms "
                f"({abs(avg_reduction):.1f}% increase)."
            )

    if full_p95_ttft > 0:
        p95_reduction = percent_reduction(
            baseline=full_p95_ttft,
            candidate=chunked_p95_ttft,
        )
        if p95_reduction >= 0:
            print(
                f"chunked prefill reduced p95 short-request TTFT "
                f"from {full_p95_ttft:.3f} ms to {chunked_p95_ttft:.3f} ms "
                f"({p95_reduction:.1f}% reduction)."
            )
        else:
            print(
                f"chunked prefill increased p95 short-request TTFT "
                f"from {full_p95_ttft:.3f} ms to {chunked_p95_ttft:.3f} ms "
                f"({abs(p95_reduction):.1f}% increase)."
            )

    if full_max_ttft > 0:
        max_reduction = percent_reduction(
            baseline=full_max_ttft,
            candidate=chunked_max_ttft,
        )
        if max_reduction >= 0:
            print(
                f"chunked prefill reduced max short-request TTFT "
                f"from {full_max_ttft:.3f} ms to {chunked_max_ttft:.3f} ms "
                f"({max_reduction:.1f}% reduction)."
            )
        else:
            print(
                f"chunked prefill increased max short-request TTFT "
                f"from {full_max_ttft:.3f} ms to {chunked_max_ttft:.3f} ms "
                f"({abs(max_reduction):.1f}% increase)."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunked-budget", type=int, default=16)
    parser.add_argument("--long-prompt-words", type=int, default=512)
    parser.add_argument("--short-prompt-words", type=int, default=8)
    parser.add_argument("--num-short-requests", type=int, default=4)
    parser.add_argument("--short-arrival-start-step", type=int, default=1)
    parser.add_argument("--short-arrival-interval-steps", type=int, default=1)
    parser.add_argument("--long-max-new-tokens", type=int, default=4)
    parser.add_argument("--short-max-new-tokens", type=int, default=8)
    parser.add_argument("--max-slots", type=int, default=4)
    parser.add_argument("--total-kv-blocks", type=int, default=768)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/chunked_prefill_ttft.csv"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_result = run_benchmark(
        mode="full",
        prefill_budget=None,
        long_prompt_words=args.long_prompt_words,
        short_prompt_words=args.short_prompt_words,
        num_short_requests=args.num_short_requests,
        short_arrival_start_step=args.short_arrival_start_step,
        short_arrival_interval_steps=args.short_arrival_interval_steps,
        long_max_new_tokens=args.long_max_new_tokens,
        short_max_new_tokens=args.short_max_new_tokens,
        max_slots=args.max_slots,
        total_kv_blocks=args.total_kv_blocks,
        block_size_tokens=args.block_size_tokens,
        max_steps=args.max_steps,
    )

    torch.cuda.empty_cache()

    chunked_result = run_benchmark(
        mode="chunked",
        prefill_budget=args.chunked_budget,
        long_prompt_words=args.long_prompt_words,
        short_prompt_words=args.short_prompt_words,
        num_short_requests=args.num_short_requests,
        short_arrival_start_step=args.short_arrival_start_step,
        short_arrival_interval_steps=args.short_arrival_interval_steps,
        long_max_new_tokens=args.long_max_new_tokens,
        short_max_new_tokens=args.short_max_new_tokens,
        max_slots=args.max_slots,
        total_kv_blocks=args.total_kv_blocks,
        block_size_tokens=args.block_size_tokens,
        max_steps=args.max_steps,
    )

    rows = [full_result, chunked_result]

    write_results(
        output_path=args.output,
        rows=rows,
    )
    print_summary(rows)

    print(f"\nWrote results to: {args.output}")


if __name__ == "__main__":
    main()