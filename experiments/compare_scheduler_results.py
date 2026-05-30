import argparse
import csv
from pathlib import Path
from typing import Any


def load_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def key_for(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["target_prompt_tokens"]),
        int(row["max_new_tokens"]),
        int(row["concurrency"]),
    )


def to_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)

    if value is None or value == "":
        return 0.0

    return float(value)


def build_index(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    return {
        key_for(row): row
        for row in rows
    }


def pct_improvement(old: float, new: float) -> float:
    if old <= 0:
        return 0.0

    return ((old - new) / old) * 100.0


def print_comparison(
    serialized_rows: list[dict[str, Any]],
    microbatch_rows: list[dict[str, Any]],
) -> None:
    serialized_by_key = build_index(serialized_rows)
    microbatch_by_key = build_index(microbatch_rows)

    common_keys = sorted(set(serialized_by_key) & set(microbatch_by_key))

    headers = [
        "prompt",
        "new",
        "conc",
        "serialized_latency",
        "microbatch_latency",
        "latency_impr_%",
        "serialized_queue",
        "microbatch_queue",
        "serialized_tok_s",
        "microbatch_tok_s",
        "tok_s_speedup",
    ]

    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for key in common_keys:
        serialized = serialized_by_key[key]
        microbatch = microbatch_by_key[key]

        prompt_tokens, max_new_tokens, concurrency = key

        serialized_latency = to_float(serialized, "client_avg_latency_seconds")
        microbatch_latency = to_float(microbatch, "client_avg_latency_seconds")

        serialized_queue = to_float(serialized, "server_avg_queue_wait_seconds")
        microbatch_queue = to_float(microbatch, "server_avg_queue_wait_seconds")

        serialized_tok_s = to_float(serialized, "approx_tokens_per_second")
        microbatch_tok_s = to_float(microbatch, "approx_tokens_per_second")

        speedup = (
            microbatch_tok_s / serialized_tok_s
            if serialized_tok_s > 0
            else 0.0
        )

        print(
            " | ".join(
                [
                    str(prompt_tokens),
                    str(max_new_tokens),
                    str(concurrency),
                    f"{serialized_latency:.4f}",
                    f"{microbatch_latency:.4f}",
                    f"{pct_improvement(serialized_latency, microbatch_latency):.1f}",
                    f"{serialized_queue:.4f}",
                    f"{microbatch_queue:.4f}",
                    f"{serialized_tok_s:.2f}",
                    f"{microbatch_tok_s:.2f}",
                    f"{speedup:.2f}x",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serialized",
        default="results/benchmark_streaming_manual_engine.csv",
    )
    parser.add_argument(
        "--microbatch",
        default="results/benchmark_streaming_static_microbatch_batch_metrics.csv",
    )
    args = parser.parse_args()

    serialized_rows = load_csv(args.serialized)
    microbatch_rows = load_csv(args.microbatch)

    print_comparison(
        serialized_rows=serialized_rows,
        microbatch_rows=microbatch_rows,
    )


if __name__ == "__main__":
    main()