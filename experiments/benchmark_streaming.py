import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


PROMPT_TOKEN_TARGETS = [32, 128]
MAX_NEW_TOKENS_VALUES = [32]
CONCURRENCY_VALUES = [1, 2, 4]
REPEATS = 3

RESULTS_PATH = "results/benchmark_streaming.csv"


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
    index = int(percentile_value * (len(sorted_values) - 1))
    return sorted_values[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }

async def fetch_metrics(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/metrics_json")
    response.raise_for_status()
    return response.json()


async def run_one_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    first_chunk_time = None
    output_parts: list[str] = []

    async with client.stream(
        "POST",
        "/generate_stream",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        },
    ) as response:
        response.raise_for_status()

        async for chunk in response.aiter_text():
            if not chunk:
                continue

            now = time.perf_counter()

            if first_chunk_time is None:
                first_chunk_time = now

            output_parts.append(chunk)

    finish_time = time.perf_counter()

    if first_chunk_time is None:
        first_chunk_time = finish_time

    generated_text = "".join(output_parts)

    return {
        "ttft_seconds": first_chunk_time - start_time,
        "latency_seconds": finish_time - start_time,
        "output_chars": len(generated_text),
    }


async def run_repeat(
    base_url: str,
    target_prompt_tokens: int,
    max_new_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    prompt = make_prompt(target_prompt_tokens)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=None,
    ) as client:
        tasks = [
            run_one_request(
                client=client,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            for _ in range(concurrency)
        ]

        return await asyncio.gather(*tasks)


async def run_case(
    base_url: str,
    target_prompt_tokens: int,
    max_new_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    # Warmup. Do not record this run.
    await run_repeat(
        base_url=base_url,
        target_prompt_tokens=target_prompt_tokens,
        max_new_tokens=max_new_tokens,
        concurrency=1,
    )

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=None,
    ) as metrics_client:
        before_metrics = await fetch_metrics(metrics_client)

    before_count = before_metrics.get("total_finished_requests", 0)

    all_request_results: list[dict[str, Any]] = []

    for _ in range(REPEATS):
        repeat_results = await run_repeat(
            base_url=base_url,
            target_prompt_tokens=target_prompt_tokens,
            max_new_tokens=max_new_tokens,
            concurrency=concurrency,
        )

        all_request_results.extend(repeat_results)

        await asyncio.sleep(0.25)

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=None,
    ) as metrics_client:
        after_metrics = await fetch_metrics(metrics_client)

    after_requests = after_metrics.get("requests", [])
    expected_new_requests = len(all_request_results)

    # The metrics endpoint returns the last 50 requests.
    # For this small benchmark, each case has at most REPEATS * concurrency requests,
    # so slicing from the end is enough and avoids relying on global request count.
    new_server_requests = after_requests[-expected_new_requests:]

    client_ttft_values = [
        result["ttft_seconds"]
        for result in all_request_results
    ]
    client_latency_values = [
        result["latency_seconds"]
        for result in all_request_results
    ]
    output_chars_values = [
        result["output_chars"]
        for result in all_request_results
    ]

    server_queue_wait_values = [
        result["queue_wait_seconds"]
        for result in new_server_requests
        if result.get("error") is None
    ]
    server_ttft_values = [
        result["ttft_seconds"]
        for result in new_server_requests
        if result.get("error") is None
    ]
    server_latency_values = [
        result["latency_seconds"]
        for result in new_server_requests
        if result.get("error") is None
    ]

    client_ttft_summary = summarize(client_ttft_values)
    client_latency_summary = summarize(client_latency_values)

    server_queue_wait_summary = summarize(server_queue_wait_values)
    server_ttft_summary = summarize(server_ttft_values)
    server_latency_summary = summarize(server_latency_values)

    total_generated_token_budget = max_new_tokens * len(all_request_results)
    total_client_latency_sum = sum(client_latency_values)

    return {
        "target_prompt_tokens": target_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "concurrency": concurrency,
        "repeats": REPEATS,
        "total_requests": len(all_request_results),

        "before_finished_requests": before_count,
        "after_finished_requests": after_metrics.get("total_finished_requests", 0),
        "server_requests_used": len(new_server_requests),

        "client_avg_ttft_seconds": client_ttft_summary["avg"],
        "client_p50_ttft_seconds": client_ttft_summary["p50"],
        "client_p95_ttft_seconds": client_ttft_summary["p95"],

        "client_avg_latency_seconds": client_latency_summary["avg"],
        "client_p50_latency_seconds": client_latency_summary["p50"],
        "client_p95_latency_seconds": client_latency_summary["p95"],

        "server_avg_queue_wait_seconds": server_queue_wait_summary["avg"],
        "server_p50_queue_wait_seconds": server_queue_wait_summary["p50"],
        "server_p95_queue_wait_seconds": server_queue_wait_summary["p95"],

        "server_avg_ttft_seconds": server_ttft_summary["avg"],
        "server_p50_ttft_seconds": server_ttft_summary["p50"],
        "server_p95_ttft_seconds": server_ttft_summary["p95"],

        "server_avg_latency_seconds": server_latency_summary["avg"],
        "server_p50_latency_seconds": server_latency_summary["p50"],
        "server_p95_latency_seconds": server_latency_summary["p95"],

        "avg_output_chars": statistics.mean(output_chars_values),
        "approx_tokens_per_second": (
            total_generated_token_budget / total_client_latency_sum
            if total_client_latency_sum > 0
            else 0.0
        ),
    }
def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "target_prompt_tokens",
        "max_new_tokens",
        "concurrency",
        "client_ttft",
        "server_queue",
        "server_ttft",
        "client_latency",
        "server_latency",
        "approx_tok_s",
    ]

    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for row in rows:
        print(
            " | ".join(
                [
                    str(row["target_prompt_tokens"]),
                    str(row["max_new_tokens"]),
                    str(row["concurrency"]),
                    f"{row['client_avg_ttft_seconds']:.4f}",
                    f"{row['server_avg_queue_wait_seconds']:.4f}",
                    f"{row['server_avg_ttft_seconds']:.4f}",
                    f"{row['client_avg_latency_seconds']:.4f}",
                    f"{row['server_avg_latency_seconds']:.4f}",
                    f"{row['approx_tokens_per_second']:.2f}",
                ]
            )
        )

def save_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "target_prompt_tokens",
        "max_new_tokens",
        "concurrency",
        "repeats",
        "total_requests",

        "before_finished_requests",
        "after_finished_requests",
        "server_requests_used",

        "client_avg_ttft_seconds",
        "client_p50_ttft_seconds",
        "client_p95_ttft_seconds",

        "client_avg_latency_seconds",
        "client_p50_latency_seconds",
        "client_p95_latency_seconds",

        "server_avg_queue_wait_seconds",
        "server_p50_queue_wait_seconds",
        "server_p95_queue_wait_seconds",

        "server_avg_ttft_seconds",
        "server_p50_ttft_seconds",
        "server_p95_ttft_seconds",

        "server_avg_latency_seconds",
        "server_p50_latency_seconds",
        "server_p95_latency_seconds",

        "avg_output_chars",
        "approx_tokens_per_second",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-path", default=RESULTS_PATH)
    args = parser.parse_args()

    rows = []

    print(f"url: {args.url}")
    print()

    for target_prompt_tokens in PROMPT_TOKEN_TARGETS:
        for max_new_tokens in MAX_NEW_TOKENS_VALUES:
            for concurrency in CONCURRENCY_VALUES:
                print(
                    f"running: target_prompt_tokens={target_prompt_tokens}, "
                    f"max_new_tokens={max_new_tokens}, "
                    f"concurrency={concurrency}",
                    flush=True,
                )

                row = await run_case(
                    base_url=args.url,
                    target_prompt_tokens=target_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    concurrency=concurrency,
                )
                rows.append(row)

                print(
                    f"finished: target_prompt_tokens={target_prompt_tokens}, "
                    f"max_new_tokens={max_new_tokens}, "
                    f"concurrency={concurrency}",
                    flush=True,
                )

    print_table(rows)
    save_csv(rows, args.output_path)

    print()
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    asyncio.run(main())