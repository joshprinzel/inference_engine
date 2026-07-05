from __future__ import annotations

import argparse
import csv
import json
import statistics
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState
from runtime.scheduling_policy import DecodeBudgetPolicy, FCFSPolicy, SchedulingPolicy

from experiments.benchmarks.workloads import build_workload





@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    num_requests: int
    max_slots: int
    max_new_tokens: int
    block_size_tokens: int
    total_kv_blocks: int
    dtype: str
    device: str
    prompt_set: str
    scheduling_policy_name: str
    max_decode_batch_size: int


@dataclass(frozen=True)
class BenchmarkResult:
    run_kind: str
    repeat_index: int

    backend: str
    num_requests: int
    max_slots: int
    max_new_tokens: int
    block_size_tokens: int
    total_kv_blocks: int
    dtype: str
    device: str
    prompt_set: str

    policy_name: str
    max_decode_batch_size: int

    total_wall_seconds: float
    tokens_generated: int
    tokens_per_second: float

    avg_queue_wait_ms: float | None
    avg_ttft_ms: float | None
    avg_decode_latency_ms: float | None
    avg_latency_ms: float | None

    decode_iterations: int
    decode_batches_built: int
    admitted_count: int
    decode_stalls: int
    kv_allocation_failures: int
    kv_oom_evictions: int
    late_admissions: int
    early_finishes: int

    backend_ms_median: float
    backend_ms_p95: float
    backend_ms_min: float
    backend_ms_max: float
    backend_ms_mean: float

    kv_peak_used_blocks: int
    kv_final_used_blocks: int
    kv_final_free_blocks: int
    kv_final_utilization: float

    all_finished: bool
    generated_text_by_request: str
    generated_tokens_by_request: str
    expected_text_by_request: str
    max_new_tokens_by_request: str
    correctness_passed: bool


def parse_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    index = int(round((p / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]





def make_engine(config: BenchmarkConfig) -> CustomLlamaDecodeEngine:
    if config.backend != "custom-cuda-paged":
        raise ValueError(
            f"Unsupported backend for v0 benchmark harness: {config.backend}. "
            "Start with custom-cuda-paged, then add hf/custom-pytorch/synthetic."
        )

    return CustomLlamaDecodeEngine(
        device=config.device,
        dtype=parse_dtype(config.dtype),
        attention_backend_name="cuda",
        total_kv_blocks=config.total_kv_blocks,
        block_size_tokens=config.block_size_tokens,
    )

def build_scheduling_policy(
        *,
        scheduling_policy_name: str,
        max_decode_batch_size: int,
) -> SchedulingPolicy:
    if scheduling_policy_name == "fcfs":
        return FCFSPolicy()
    
    if scheduling_policy_name == "decode_budget":
        return DecodeBudgetPolicy(max_decode_batch_size=max_decode_batch_size)
    
    raise ValueError(f"Unsupported scheduling policy: {scheduling_policy_name}")


def run_single_benchmark(
        config: BenchmarkConfig,
        run_kind: str = "measured",
        repeat_index: int = 0) -> BenchmarkResult:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but CUDA is not available")

    if config.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    engine = make_engine(config)

    request_queue = RequestQueue()
    metrics_store = MetricsStore()
    kv_block_manager = KVBlockManager(
        total_blocks=config.total_kv_blocks,
        block_size_tokens=config.block_size_tokens,
    )

    scheduling_policy = build_scheduling_policy(
        scheduling_policy_name=config.scheduling_policy_name,
        max_decode_batch_size=config.max_decode_batch_size
    )

    scheduler = EngineScheduler(
        decode_engine=engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=config.max_slots,
        scheduling_policy=scheduling_policy
    )

    request_specs = build_workload(
        prompt_set=config.prompt_set,
        num_requests=config.num_requests,
        max_new_tokens=config.max_new_tokens
    )

    expected_text_by_request: dict[str, str] = {}
    max_new_tokens_by_request: dict[str, int] = {}

    for request_spec in request_specs:
        request_state = RequestState(
            request_id=request_spec.request_id,
            prompt=request_spec.prompt,
            max_new_tokens=request_spec.max_new_tokens,
        )

        request_queue.put(request_state)

        # We only require the prefix because tokenization/newline tails can vary
        # once max_new_tokens changes.
        expected_text_by_request[request_spec.request_id] = request_spec.expected_prefix
        max_new_tokens_by_request[request_spec.request_id] = request_spec.max_new_tokens

    backend_ms_values: list[float] = []
    kv_used_blocks_values: list[int] = []

    max_request_new_tokens = max(
        request_spec.max_new_tokens for request_spec in request_specs
    )

    slot_waves = math.ceil(config.num_requests / max(1, config.max_slots))
    max_steps = (
        max_request_new_tokens * slot_waves + config.num_requests + 16
    )

    if config.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(max_steps):
        scheduler.step()

        if scheduler.last_backend_ms is not None:
            backend_ms_values.append(float(scheduler.last_backend_ms))

        snapshot = scheduler.snapshot()
        kv_used_blocks_values.append(int(snapshot["kv_used_blocks"]))

        if len(scheduler.finished) == config.num_requests:
            break

    if config.device == "cuda":
        torch.cuda.synchronize()

    total_wall_seconds = time.perf_counter() - start

    final_snapshot = scheduler.snapshot()
    kv_snapshot = scheduler.kv_block_manager.snapshot()
    metrics_snapshot = metrics_store.snapshot()

    generated_text_by_request = {
        request.request_id: request.generated_text
        for request in scheduler.finished
    }

    generated_tokens_by_request = {
        request.request_id: int(request.generated_tokens)
        for request in scheduler.finished
    }

    correctness_by_request = {
        request_id: generated_text_by_request.get(request_id, "").startswith(expected_prefix)
        for request_id, expected_prefix in expected_text_by_request.items()
    }

    correctness_passed = all(correctness_by_request.values())
    all_finished = len(scheduler.finished) == config.num_requests

    tokens_generated = int(scheduler.tokens_generated)
    tokens_per_second = (
        tokens_generated / total_wall_seconds if total_wall_seconds > 0 else 0.0
    )

    return BenchmarkResult(
        run_kind=run_kind,
        repeat_index=repeat_index,
        backend=config.backend,
        num_requests=config.num_requests,
        max_slots=config.max_slots,
        max_new_tokens=config.max_new_tokens,
        block_size_tokens=config.block_size_tokens,
        total_kv_blocks=config.total_kv_blocks,
        dtype=config.dtype,
        device=config.device,
        prompt_set=config.prompt_set,

        policy_name=str(final_snapshot["policy_name"]),
        max_decode_batch_size=config.max_decode_batch_size,

        total_wall_seconds=total_wall_seconds,
        tokens_generated=tokens_generated,
        tokens_per_second=tokens_per_second,

        avg_queue_wait_ms=metrics_snapshot.get("avg_queue_wait_ms"),
        avg_ttft_ms=metrics_snapshot.get("avg_ttft_ms"),
        avg_decode_latency_ms=metrics_snapshot.get("avg_decode_latency_ms"),
        avg_latency_ms=metrics_snapshot.get("avg_latency_ms"),

        decode_iterations=int(final_snapshot["decode_iterations"]),
        decode_batches_built=int(final_snapshot["decode_batches_built"]),
        admitted_count=int(final_snapshot["admitted_count"]),
        decode_stalls=int(final_snapshot["decode_stalls"]),
        kv_allocation_failures=int(final_snapshot["kv_allocation_failures"]),
        kv_oom_evictions=int(final_snapshot["kv_oom_evictions"]),
        late_admissions=int(final_snapshot["late_admissions"]),
        early_finishes=int(final_snapshot["early_finishes"]),
        backend_ms_median=statistics.median(backend_ms_values)
        if backend_ms_values
        else 0.0,
        backend_ms_p95=percentile(backend_ms_values, 95.0),
        backend_ms_min=min(backend_ms_values) if backend_ms_values else 0.0,
        backend_ms_max=max(backend_ms_values) if backend_ms_values else 0.0,
        backend_ms_mean=statistics.mean(backend_ms_values)
        if backend_ms_values
        else 0.0,
        kv_peak_used_blocks=max(kv_used_blocks_values) if kv_used_blocks_values else 0,
        kv_final_used_blocks=int(kv_snapshot["used_blocks"]),
        kv_final_free_blocks=int(kv_snapshot["free_blocks"]),
        kv_final_utilization=float(kv_snapshot["utilization"]),
        all_finished=all_finished,
        generated_text_by_request=json.dumps(
            generated_text_by_request,
            sort_keys=True,
        ),
        expected_text_by_request=json.dumps(
            expected_text_by_request,
            sort_keys=True,
        ),
        max_new_tokens_by_request=json.dumps(
            max_new_tokens_by_request,
            sort_keys=True
        ),
        generated_tokens_by_request=json.dumps(
            generated_tokens_by_request,
            sort_keys=True,
        ),
        correctness_passed=correctness_passed,
    )


def write_jsonl(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot write empty benchmark rows")

    fieldnames = list(asdict(rows[0]).keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def config_key(row: BenchmarkResult) -> tuple[Any, ...]:
    return (
        row.backend,
        row.num_requests,
        row.max_slots,
        row.max_new_tokens,
        row.block_size_tokens,
        row.total_kv_blocks,
        row.dtype,
        row.device,
        row.prompt_set,
        row.policy_name,
        row.max_decode_batch_size,
    )


def median_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))

def median_optional_float(values: list[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return float(statistics.median(present_values))


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(statistics.median(values))


def write_summary(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    measured_rows = [row for row in rows if row.run_kind == "measured"]

    grouped: dict[tuple[Any, ...], list[BenchmarkResult]] = {}

    for row in measured_rows:
        grouped.setdefault(config_key(row), []).append(row)

    lines = [
        "# Runtime Benchmark Summary",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        f"Measured configurations: `{len(grouped)}`",
        f"Measured rows: `{len(measured_rows)}`",
        "",
        "| backend | policy | requests | slots | new tokens | block size | repeats | tok/s median | tok/s min | tok/s max | TTFT ms median | latency ms median | backend ms median | backend ms p95 median | peak KV blocks | correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, group in sorted(grouped.items(), key=lambda item: item[0]):
        representative = group[0]

        tokens_per_second_values = [row.tokens_per_second for row in group]
        backend_ms_median_values = [row.backend_ms_median for row in group]
        backend_ms_p95_values = [row.backend_ms_p95 for row in group]
        kv_peak_values = [row.kv_peak_used_blocks for row in group]
        ttft_ms_values = [row.avg_ttft_ms for row in group]
        latency_ms_values = [row.avg_latency_ms for row in group]

        correctness_passed = all(row.correctness_passed for row in group)

        lines.append(
            "| "
            f"{representative.backend} | "
            f"{representative.policy_name} | "
            f"{representative.num_requests} | "
            f"{representative.max_slots} | "
            f"{representative.max_new_tokens} | "
            f"{representative.block_size_tokens} | "
            f"{len(group)} | "
            f"{median_float(tokens_per_second_values):.2f} | "
            f"{min(tokens_per_second_values):.2f} | "
            f"{max(tokens_per_second_values):.2f} | "
            f"{format_optional_float(median_optional_float(ttft_ms_values))} | "
            f"{format_optional_float(median_optional_float(latency_ms_values))} | "
            f"{median_float(backend_ms_median_values):.3f} | "
            f"{median_float(backend_ms_p95_values):.3f} | "
            f"{median_int(kv_peak_values)} | "
            f"{correctness_passed} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Warmup rows are written to JSONL/CSV but excluded from this summary.",
            "- Summary rows aggregate repeated measured runs by benchmark configuration.",
            "- `total_wall_seconds` is end-to-end scheduler wall time.",
            "- `backend_ms_*` comes from the decode engine's backend timing and includes Python/model/backend work inside `decode_step`.",
            "- `kv_peak_used_blocks` is useful for graphing KV pressure under concurrency.",
            "- `correctness_passed` checks generated text prefixes for the benchmark prompts.",
            "- `avg_ttft_ms` is average request time-to-first-token for finished successful requests.",
            "- `avg_latency_ms` is average end-to-end request latency for finished successful requests.",
            "- `policy_name` identifies the scheduler policy used for admission/decode selection.",
            "- `capitals` is the fixed-length correctness/control workload.",
            "- `mixed_short_long` varies per-request decode length while keeping prompts correctness-checkable.",
            "- Slot pressure is created by running with `num_requests > max_slots`.",
            "- `max_new_tokens_by_request` records the per-request decode limit used by synthetic workloads.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_configs(args: argparse.Namespace) -> list[BenchmarkConfig]:
    num_requests_values = parse_int_list(args.num_requests)
    max_new_tokens_values = parse_int_list(args.max_new_tokens)
    block_size_values = parse_int_list(args.block_size_tokens)

    configs: list[BenchmarkConfig] = []

    for num_requests in num_requests_values:
        for max_new_tokens in max_new_tokens_values:
            for block_size_tokens in block_size_values:
                max_slots = args.max_slots or num_requests

                configs.append(
                    BenchmarkConfig(
                        backend=args.backend,
                        num_requests=num_requests,
                        max_slots=max_slots,
                        max_new_tokens=max_new_tokens,
                        block_size_tokens=block_size_tokens,
                        total_kv_blocks=args.total_kv_blocks,
                        dtype=args.dtype,
                        device=args.device,
                        prompt_set=args.prompt_set,
                        scheduling_policy_name=args.scheduling_policy,
                        max_decode_batch_size=args.max_decode_batch_size,
                    )
                )

    return configs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        type=str,
        default="custom-cuda-paged",
        choices=["custom-cuda-paged"],
    )
    parser.add_argument(
        "--num-requests",
        type=str,
        default="1,2,4",
        help="Comma-separated request counts, e.g. 1,2,4",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="If omitted, max_slots=num_requests for each run",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=str,
        default="8,16,32",
        help="Comma-separated max_new_tokens values",
    )
    parser.add_argument(
        "--block-size-tokens",
        type=str,
        default="16",
        help="Comma-separated block sizes, e.g. 4,8,16,32",
    )

    parser.add_argument(
        "--scheduling-policy",
        type=str,
        default="fcfs",
        choices=["fcfs", "decode_budget"],
    )
    parser.add_argument(
        "--max-decode-batch-size",
        type=int,
        default=4,
        help="Used only when --scheduling-policy=decode_budget",
    )
    parser.add_argument("--total-kv-blocks", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt-set", type=str, default="capitals", choices=["capitals", "mixed_short_long"])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks"),
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = f"runtime_benchmark_{timestamp}"

    jsonl_path = args.output_dir / f"{output_stem}.jsonl"
    csv_path = args.output_dir / f"{output_stem}.csv"
    summary_path = args.output_dir / f"{output_stem}_summary.md"

    configs = build_configs(args)

    rows: list[BenchmarkResult] = []

    total_runs = len(configs) * (args.warmup_runs + args.repeat_runs)
    run_counter = 0

    for config_index, config in enumerate(configs, start=1):
        print(f"[config {config_index}/{len(configs)}] {config}")

        for warmup_index in range(args.warmup_runs):
            run_counter += 1
            print(f"  [run {run_counter}/{total_runs}] warmup {warmup_index}")

            row = run_single_benchmark(
                config=config,
                run_kind="warmup",
                repeat_index=warmup_index,
            )
            rows.append(row)

            print(
                "    "
                f"policy={row.policy_name} "
                f"tokens_per_second={row.tokens_per_second:.2f} "
                f"avg_ttft_ms={format_optional_float(row.avg_ttft_ms)} "
                f"avg_latency_ms={format_optional_float(row.avg_latency_ms)} "
                f"backend_ms_median={row.backend_ms_median:.3f} "
                f"backend_ms_p95={row.backend_ms_p95:.3f} "
                f"kv_peak_used_blocks={row.kv_peak_used_blocks} "
                f"correct={row.correctness_passed}"
            )

        for repeat_index in range(args.repeat_runs):
            run_counter += 1
            print(f"  [run {run_counter}/{total_runs}] measured {repeat_index}")

            row = run_single_benchmark(
                config=config,
                run_kind="measured",
                repeat_index=repeat_index,
            )
            rows.append(row)

            print(
                "    "
                f"policy={row.policy_name} "
                f"tokens_per_second={row.tokens_per_second:.2f} "
                f"avg_ttft_ms={format_optional_float(row.avg_ttft_ms)} "
                f"avg_latency_ms={format_optional_float(row.avg_latency_ms)} "
                f"backend_ms_median={row.backend_ms_median:.3f} "
                f"backend_ms_p95={row.backend_ms_p95:.3f} "
                f"kv_peak_used_blocks={row.kv_peak_used_blocks} "
                f"correct={row.correctness_passed}"
            )
    

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    write_summary(summary_path, rows)

    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()