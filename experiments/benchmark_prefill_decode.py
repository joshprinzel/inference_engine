import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model_runner import ModelRunner


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT_TOKEN_TARGETS = [32, 128, 512, 1024]
MAX_NEW_TOKENS_VALUES = [32, 64, 128]
REPEATS = 5

RESULTS_PATH = "results/benchmark_prefill_decode.csv"


def make_prompt(target_tokens: int) -> str:
    base = (
        "In transformer inference, explain KV cache, prefill, decode, "
        "continuous batching, and memory pressure. "
    )

    return base * max(1, target_tokens // 16)

def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = int(percentile_value * (len(sorted_values)-1))
    return sorted_values[index]



def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values,0.50),
        "p95": percentile(values,0.95),
    }


def run_case(
    runner: ModelRunner,
    target_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = make_prompt(target_prompt_tokens)

    # Warmup. Do not record this run.
    runner.benchmark_prefill_decode(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    prompt_tokens_values = []
    generated_tokens_values = []
    prefill_times = []
    decode_times = []
    total_times = []
    decode_tps_values = []
    total_tps_values = []

    for _ in range(REPEATS):
        result = runner.benchmark_prefill_decode(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        prompt_tokens_values.append(result["prompt_tokens"])
        generated_tokens_values.append(result["generated_tokens"])
        prefill_times.append(result["prefill_time_seconds"])
        decode_times.append(result["decode_time_seconds"])
        total_times.append(result["total_time_seconds"])
        decode_tps_values.append(result["decode_tokens_per_second"])
        total_tps_values.append(result["total_tokens_per_second"])

        time.sleep(0.25)

    prefill_summary = summarize(prefill_times)
    decode_summary = summarize(decode_times)
    total_summary = summarize(total_times)
    decode_tps_summary = summarize(decode_tps_values)
    total_tps_summary = summarize(total_tps_values)


    return {
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens": int(statistics.mean(prompt_tokens_values)),
        "max_new_tokens": max_new_tokens,
        "avg_generated_tokens": statistics.mean(generated_tokens_values),

        "avg_prefill_time_seconds": prefill_summary["avg"],
        "p50_prefill_time_seconds": prefill_summary["p50"],
        "p95_prefill_time_seconds": prefill_summary["p95"],

        "avg_decode_time_seconds": decode_summary["avg"],
        "p50_decode_time_seconds": decode_summary["p50"],
        "p95_decode_time_seconds": decode_summary["p95"],

        "avg_total_time_seconds": total_summary["avg"],
        "p50_total_time_seconds": total_summary["p50"],
        "p95_total_time_seconds": total_summary["p95"],
        "min_total_time_seconds": total_summary["min"],
        "max_total_time_seconds": total_summary["max"],

        "avg_decode_tokens_per_second": decode_tps_summary["avg"],
        "avg_total_tokens_per_second": total_tps_summary["avg"],
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "target_prompt_tokens",
        "prompt_tokens",
        "max_new_tokens",
        "avg_prefill",
        "p95_prefill",
        "avg_decode",
        "p95_decode",
        "avg_total",
        "p95_total",
        "decode_tok_s",
        "total_tok_s",
    ]

    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for row in rows:
        print(
            " | ".join(
                [
                    str(row["target_prompt_tokens"]),
                    str(row["prompt_tokens"]),
                    str(row["max_new_tokens"]),
                    f"{row['avg_prefill_time_seconds']:.4f}",
                    f"{row['p95_prefill_time_seconds']:.4f}",
                    f"{row['avg_decode_time_seconds']:.4f}",
                    f"{row['p95_decode_time_seconds']:.4f}",
                    f"{row['avg_total_time_seconds']:.4f}",
                    f"{row['p95_total_time_seconds']:.4f}",
                    f"{row['avg_decode_tokens_per_second']:.2f}",
                    f"{row['avg_total_tokens_per_second']:.2f}",
                ]
            )
        )


def save_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "target_prompt_tokens",
        "prompt_tokens",
        "max_new_tokens",
        "avg_generated_tokens",

        "avg_prefill_time_seconds",
        "p50_prefill_time_seconds",
        "p95_prefill_time_seconds",

        "avg_decode_time_seconds",
        "p50_decode_time_seconds",
        "p95_decode_time_seconds",

        "avg_total_time_seconds",
        "p50_total_time_seconds",
        "p95_total_time_seconds",
        "min_total_time_seconds",
        "max_total_time_seconds",

        "avg_decode_tokens_per_second",
        "avg_total_tokens_per_second",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    runner = ModelRunner(model_name=MODEL_NAME)

    print(f"model: {runner.model_name}")
    print(f"device: {runner.device}")
    print()

    rows = []

    for target_prompt_tokens in PROMPT_TOKEN_TARGETS:
        for max_new_tokens in MAX_NEW_TOKENS_VALUES:
            row = run_case(
                runner=runner,
                target_prompt_tokens=target_prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
            rows.append(row)

    print_table(rows)
    save_csv(rows, RESULTS_PATH)

    print()
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()