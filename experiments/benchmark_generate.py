import statistics
import time
from typing import Any

import requests
import csv
from pathlib import Path

SERVER_URL = "http://localhost:8000/generate"

PROMPT_TOKEN_TARGETS = [32,128,512,1024]
MAX_NEW_TOKENS_VALUES = [32,64,128]
REPEATS = 5

def make_prompt(target_tokens:int) -> str:
    base = (
        "In transformer inference, explain KV cache, prefill, decode, "
        "continuous batching, and memory pressure. "
    )
    return base * max(1, target_tokens // 16)

def call_generate(prompt: str, max_new_tokens: int) -> dict[str, Any]:
    response = requests.post(
        SERVER_URL,
        json={
            "prompt":prompt,
            "max_new_tokens": max_new_tokens,
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()

def summarize(values: list[float]) -> dict[str,float]:
    return{
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def run_case(prompt: str, max_new_tokens: int) -> dict[str,Any]:
    #Warmup: avoids first-call overhead affecting measurements
    call_generate(prompt=prompt, max_new_tokens=max_new_tokens)
    latencies = []
    tokens_per_second_values = []
    generated_tokens_values = []
    prompt_tokens_values = []

    for _ in range(REPEATS):
        result = call_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

        latencies.append(result["latency_seconds"])
        tokens_per_second_values.append(result["tokens_per_second"])
        generated_tokens_values.append(result["generated_tokens"])
        prompt_tokens_values.append(result["prompt_tokens"])

        #Small pause so repeated requests are easier to read in server logs
        time.sleep(0.25)

    latency_summary = summarize(latencies)
    tps_summary = summarize(tokens_per_second_values)

    return {
        "prompt_tokens": int(statistics.mean(prompt_tokens_values)),
        "max_new_tokens": max_new_tokens,
        "avg_generated_tokens": statistics.mean(generated_tokens_values),
        "avg_latency_seconds": latency_summary["avg"],
        "min_latency_seconds": latency_summary["min"],
        "max_latency_seconds": latency_summary["max"],
        "avg_tokens_per_second": tps_summary["avg"],
    }

def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "target_prompt_tokens",
        "prompt_tokens",
        "max_new_tokens",
        "avg_generated_tokens",
        "avg_latency_seconds",
        "min_latency_seconds",
        "max_latency_seconds",
        "avg_tokens_per_second",
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
                    f"{row['avg_generated_tokens']:.1f}",
                    f"{row['avg_latency_seconds']:.3f}",
                    f"{row['min_latency_seconds']:.3f}",
                    f"{row['max_latency_seconds']:.3f}",
                    f"{row['avg_tokens_per_second']:.2f}",
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
        "avg_latency_seconds",
        "min_latency_seconds",
        "max_latency_seconds",
        "avg_tokens_per_second",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    rows = []

    for target_prompt_tokens in PROMPT_TOKEN_TARGETS:
        prompt = make_prompt(target_prompt_tokens)

        for max_new_tokens in MAX_NEW_TOKENS_VALUES:
            row = run_case(
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            row["target_prompt_tokens"] = target_prompt_tokens
            rows.append(row)

    print_table(rows)
    save_csv(rows, "results/benchmark_generate.csv")
    print()
    print("Saved results to results/benchmark_generate.csv")


if __name__ == "__main__":
    main()