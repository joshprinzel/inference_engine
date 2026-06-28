from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BenchmarkRow:
    run_kind: str
    backend: str
    num_requests: int
    max_slots: int
    max_new_tokens: int
    block_size_tokens: int
    total_kv_blocks: int
    dtype: str
    device: str
    prompt_set: str
    tokens_per_second: float
    backend_ms_median: float
    backend_ms_p95: float
    kv_peak_used_blocks: int
    correctness_passed: bool


@dataclass(frozen=True)
class AggregatedRow:
    backend: str
    num_requests: int
    max_slots: int
    max_new_tokens: int
    block_size_tokens: int
    total_kv_blocks: int
    dtype: str
    device: str
    prompt_set: str
    repeats: int
    tokens_per_second_median: float
    tokens_per_second_min: float
    tokens_per_second_max: float
    backend_ms_median_median: float
    backend_ms_p95_median: float
    kv_peak_used_blocks_median: int
    correctness_passed: bool


def parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes"}


def read_rows(csv_path: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            rows.append(
                BenchmarkRow(
                    run_kind=raw.get("run_kind", "measured"),
                    backend=raw["backend"],
                    num_requests=int(raw["num_requests"]),
                    max_slots=int(raw["max_slots"]),
                    max_new_tokens=int(raw["max_new_tokens"]),
                    block_size_tokens=int(raw["block_size_tokens"]),
                    total_kv_blocks=int(raw["total_kv_blocks"]),
                    dtype=raw["dtype"],
                    device=raw["device"],
                    prompt_set=raw["prompt_set"],
                    tokens_per_second=float(raw["tokens_per_second"]),
                    backend_ms_median=float(raw["backend_ms_median"]),
                    backend_ms_p95=float(raw["backend_ms_p95"]),
                    kv_peak_used_blocks=int(raw["kv_peak_used_blocks"]),
                    correctness_passed=parse_bool(raw["correctness_passed"]),
                )
            )

    return rows


def group_key(row: BenchmarkRow) -> tuple[Any, ...]:
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
    )


def median_int(values: list[int]) -> int:
    return int(statistics.median(values))


def aggregate_rows(rows: list[BenchmarkRow]) -> list[AggregatedRow]:
    measured_rows = [row for row in rows if row.run_kind == "measured"]

    grouped: dict[tuple[Any, ...], list[BenchmarkRow]] = defaultdict(list)

    for row in measured_rows:
        grouped[group_key(row)].append(row)

    aggregated: list[AggregatedRow] = []

    for _, group in grouped.items():
        first = group[0]

        tokens_per_second_values = [row.tokens_per_second for row in group]
        backend_ms_median_values = [row.backend_ms_median for row in group]
        backend_ms_p95_values = [row.backend_ms_p95 for row in group]
        kv_peak_values = [row.kv_peak_used_blocks for row in group]

        aggregated.append(
            AggregatedRow(
                backend=first.backend,
                num_requests=first.num_requests,
                max_slots=first.max_slots,
                max_new_tokens=first.max_new_tokens,
                block_size_tokens=first.block_size_tokens,
                total_kv_blocks=first.total_kv_blocks,
                dtype=first.dtype,
                device=first.device,
                prompt_set=first.prompt_set,
                repeats=len(group),
                tokens_per_second_median=float(statistics.median(tokens_per_second_values)),
                tokens_per_second_min=min(tokens_per_second_values),
                tokens_per_second_max=max(tokens_per_second_values),
                backend_ms_median_median=float(statistics.median(backend_ms_median_values)),
                backend_ms_p95_median=float(statistics.median(backend_ms_p95_values)),
                kv_peak_used_blocks_median=median_int(kv_peak_values),
                correctness_passed=all(row.correctness_passed for row in group),
            )
        )

    return sorted(
        aggregated,
        key=lambda row: (
            row.backend,
            row.block_size_tokens,
            row.max_new_tokens,
            row.num_requests,
        ),
    )


def filter_rows(
    rows: list[AggregatedRow],
    block_size_tokens: int | None,
    max_new_tokens: int | None,
) -> list[AggregatedRow]:
    filtered = rows

    if block_size_tokens is not None:
        filtered = [
            row for row in filtered if row.block_size_tokens == block_size_tokens
        ]

    if max_new_tokens is not None:
        filtered = [row for row in filtered if row.max_new_tokens == max_new_tokens]

    return filtered


def group_by_max_new_tokens(
    rows: list[AggregatedRow],
) -> dict[int, list[AggregatedRow]]:
    grouped: dict[int, list[AggregatedRow]] = defaultdict(list)

    for row in rows:
        grouped[row.max_new_tokens].append(row)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda row: row.num_requests)

    return dict(sorted(grouped.items()))


def group_by_block_size(
    rows: list[AggregatedRow],
) -> dict[int, list[AggregatedRow]]:
    grouped: dict[int, list[AggregatedRow]] = defaultdict(list)

    for row in rows:
        grouped[row.block_size_tokens].append(row)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda row: row.num_requests)

    return dict(sorted(grouped.items()))


def save_line_plot_by_max_new_tokens(
    rows: list[AggregatedRow],
    output_path: Path,
    y_getter,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = group_by_max_new_tokens(rows)

    plt.figure(figsize=(8, 5))

    for max_new_tokens, group in grouped.items():
        x = [row.num_requests for row in group]
        y = [y_getter(row) for row in group]

        plt.plot(
            x,
            y,
            marker="o",
            label=f"max_new_tokens={max_new_tokens}",
        )

    plt.xlabel("Number of requests")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_line_plot_by_block_size(
    rows: list[AggregatedRow],
    output_path: Path,
    y_getter,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = group_by_block_size(rows)

    plt.figure(figsize=(8, 5))

    for block_size_tokens, group in grouped.items():
        x = [row.num_requests for row in group]
        y = [y_getter(row) for row in group]

        plt.plot(
            x,
            y,
            marker="o",
            label=f"block_size={block_size_tokens}",
        )

    plt.xlabel("Number of requests")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_block_size_sensitivity_plot(
    rows: list[AggregatedRow],
    output_path: Path,
    y_getter,
    ylabel: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_by_requests: dict[int, list[AggregatedRow]] = defaultdict(list)

    for row in rows:
        grouped_by_requests[row.num_requests].append(row)

    for key in grouped_by_requests:
        grouped_by_requests[key] = sorted(
            grouped_by_requests[key],
            key=lambda row: row.block_size_tokens,
        )

    plt.figure(figsize=(8, 5))

    for num_requests, group in sorted(grouped_by_requests.items()):
        x = [row.block_size_tokens for row in group]
        y = [y_getter(row) for row in group]

        plt.plot(
            x,
            y,
            marker="o",
            label=f"requests={num_requests}",
        )

    plt.xlabel("Block size tokens")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_plot_summary(
    output_path: Path,
    csv_path: Path,
    plots: list[Path],
    rows: list[AggregatedRow],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Runtime Benchmark Plots",
        "",
        f"Source CSV: `{csv_path}`",
        "",
        f"Aggregated measured configurations: `{len(rows)}`",
        "",
        "## Generated plots",
        "",
    ]

    for plot in plots:
        lines.append(f"- `{plot}`")

    lines.extend(
        [
            "",
            "## Plot interpretation checklist",
            "",
            "- `tokens_per_second_vs_requests.png`: shows end-to-end serving throughput as active requests increase.",
            "- `backend_ms_median_vs_requests.png`: shows median decode engine latency and exposes per-request loop scaling.",
            "- `backend_ms_p95_vs_requests.png`: shows tail latency behavior.",
            "- `kv_peak_blocks_vs_requests.png`: shows KV cache pressure under concurrency.",
            "- Block-size sensitivity plots show how block granularity affects throughput and KV allocation.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV produced by experiments.benchmarks.bench_runtime",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks/plots"),
    )
    parser.add_argument(
        "--block-size-tokens",
        type=int,
        default=None,
        help="Optional block-size filter for request-scaling plots.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max-new-token filter for block-size sensitivity plots.",
    )

    args = parser.parse_args()

    raw_rows = read_rows(args.csv)
    aggregated_rows = aggregate_rows(raw_rows)

    if not aggregated_rows:
        raise RuntimeError("No measured benchmark rows found")

    request_scaling_rows = filter_rows(
        aggregated_rows,
        block_size_tokens=args.block_size_tokens,
        max_new_tokens=None,
    )

    if not request_scaling_rows:
        raise RuntimeError("No rows available for request-scaling plots")

    plots: list[Path] = []

    tokens_plot = args.output_dir / "tokens_per_second_vs_requests.png"
    save_line_plot_by_max_new_tokens(
        rows=request_scaling_rows,
        output_path=tokens_plot,
        y_getter=lambda row: row.tokens_per_second_median,
        ylabel="Tokens / second",
        title="Throughput vs number of requests",
    )
    plots.append(tokens_plot)

    backend_median_plot = args.output_dir / "backend_ms_median_vs_requests.png"
    save_line_plot_by_max_new_tokens(
        rows=request_scaling_rows,
        output_path=backend_median_plot,
        y_getter=lambda row: row.backend_ms_median_median,
        ylabel="Backend median latency (ms)",
        title="Backend median latency vs number of requests",
    )
    plots.append(backend_median_plot)

    backend_p95_plot = args.output_dir / "backend_ms_p95_vs_requests.png"
    save_line_plot_by_max_new_tokens(
        rows=request_scaling_rows,
        output_path=backend_p95_plot,
        y_getter=lambda row: row.backend_ms_p95_median,
        ylabel="Backend p95 latency (ms)",
        title="Backend p95 latency vs number of requests",
    )
    plots.append(backend_p95_plot)

    kv_plot = args.output_dir / "kv_peak_blocks_vs_requests.png"
    save_line_plot_by_max_new_tokens(
        rows=request_scaling_rows,
        output_path=kv_plot,
        y_getter=lambda row: row.kv_peak_used_blocks_median,
        ylabel="Peak used KV blocks",
        title="Peak KV blocks vs number of requests",
    )
    plots.append(kv_plot)

    block_sensitivity_rows = aggregated_rows
    if args.max_new_tokens is not None:
        block_sensitivity_rows = [
            row
            for row in block_sensitivity_rows
            if row.max_new_tokens == args.max_new_tokens
        ]

    unique_block_sizes = {
        row.block_size_tokens for row in block_sensitivity_rows
    }

    if len(unique_block_sizes) > 1:
        block_tokens_plot = args.output_dir / "tokens_per_second_vs_block_size.png"
        save_block_size_sensitivity_plot(
            rows=block_sensitivity_rows,
            output_path=block_tokens_plot,
            y_getter=lambda row: row.tokens_per_second_median,
            ylabel="Tokens / second",
            title="Throughput vs block size",
        )
        plots.append(block_tokens_plot)

        block_backend_plot = args.output_dir / "backend_ms_median_vs_block_size.png"
        save_block_size_sensitivity_plot(
            rows=block_sensitivity_rows,
            output_path=block_backend_plot,
            y_getter=lambda row: row.backend_ms_median_median,
            ylabel="Backend median latency (ms)",
            title="Backend median latency vs block size",
        )
        plots.append(block_backend_plot)

    summary_path = args.output_dir / "plot_summary.md"
    write_plot_summary(
        output_path=summary_path,
        csv_path=args.csv,
        plots=plots,
        rows=aggregated_rows,
    )

    print(f"Read CSV: {args.csv}")
    print(f"Aggregated measured rows: {len(aggregated_rows)}")
    for plot in plots:
        print(f"Wrote plot: {plot}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()