import argparse
import csv
import statistics
from pathlib import Path
from typing import Any


def load_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    return rows


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(percentile_value * (len(sorted_values) - 1))
    return sorted_values[index]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "avg": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def group_by_batch_size(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}

    for row in rows:
        batch_size = int(row["batch_size"])
        groups.setdefault(batch_size, []).append(row)

    return groups


def values_for(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def print_summary(rows: list[dict[str, Any]], drop_first: bool) -> None:
    if drop_first and rows:
        rows = rows[1:]

    groups = group_by_batch_size(rows)

    headers = [
        "batch_size",
        "batches",
        "avg_prefill",
        "avg_decode",
        "p95_decode",
        "avg_total",
        "avg_batch_tok_s",
        "p50_batch_tok_s",
        "p95_batch_tok_s",
    ]

    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for batch_size in sorted(groups):
        group = groups[batch_size]

        prefill = summarize(values_for(group, "prefill_time_seconds"))
        decode = summarize(values_for(group, "decode_time_seconds_total"))
        total = summarize(values_for(group, "total_time_seconds"))
        tok_s = summarize(values_for(group, "batch_tokens_per_second"))

        print(
            " | ".join(
                [
                    str(batch_size),
                    str(len(group)),
                    f"{prefill['avg']:.4f}",
                    f"{decode['avg']:.4f}",
                    f"{decode['p95']:.4f}",
                    f"{total['avg']:.4f}",
                    f"{tok_s['avg']:.2f}",
                    f"{tok_s['p50']:.2f}",
                    f"{tok_s['p95']:.2f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="results/benchmark_streaming_static_microbatch_engine_batches.csv",
    )
    parser.add_argument(
        "--drop-first",
        action="store_true",
        help="Drop the first batch as warmup.",
    )
    args = parser.parse_args()

    rows = load_csv(args.input_path)

    print(f"input: {args.input_path}")
    print(f"rows: {len(rows)}")
    print(f"drop_first: {args.drop_first}")
    print()

    print_summary(rows, drop_first=args.drop_first)


if __name__ == "__main__":
    main()