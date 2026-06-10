import asyncio
import time
from typing import Any

import httpx


async def run_one_request(
    client: httpx.AsyncClient,
    request_id: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = (
        f"Request {request_id}: Explain KV cache memory allocation in LLM serving. "
        f"Give a detailed but concise explanation."
    )

    start = time.perf_counter()
    first = None
    chunks: list[str] = []

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

            if first is None:
                first = now

            chunks.append(chunk)

    end = time.perf_counter()

    return {
        "request_id": request_id,
        "max_new_tokens": max_new_tokens,
        "ttft_seconds": (first or end) - start,
        "latency_seconds": end - start,
        "output_chars": len("".join(chunks)),
    }


async def poll_metrics(client: httpx.AsyncClient, duration_seconds: float = 5.0) -> None:
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        response = await client.get("/metrics_json")
        response.raise_for_status()
        data = response.json()
        engine = data["engine"]
        kv_cache = engine["kv_cache"]

        print(
            "metrics | "
            f"waiting={engine['waiting']} "
            f"active={engine['active']} "
            f"used_blocks={kv_cache['used_blocks']} "
            f"free_blocks={kv_cache['free_blocks']} "
            f"kv_active_requests={kv_cache['active_requests']}"
        )

        await asyncio.sleep(0.25)

async def print_final_metrics(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics_json")
    response.raise_for_status()
    data = response.json()

    engine = data["engine"]
    kv_cache = engine["kv_cache"]

    print()
    print("engine pressure metrics")
    print("---")
    print(f"decode_iterations: {engine.get('decode_iterations', engine.get('decode_steps'))}")
    print(f"decode_stalls: {engine.get('decode_stalls')}")
    print(f"kv_allocation_failures: {engine.get('kv_allocation_failures')}")
    print(f"kv_oom_evictions: {engine.get('kv_oom_evictions')}")
    print(f"used_blocks: {kv_cache['used_blocks']}")
    print(f"free_blocks: {kv_cache['free_blocks']}")
    print(f"active_requests: {kv_cache['active_requests']}")

    print()
    print("finished request status")
    print("---")
    print("request_id | max_new_tokens | generated_tokens | status | error")
    print("--- | --- | --- | --- | ---")

    for request in data["requests"]:
        print(
            f"{request['request_id']} | "
            f"{request['max_new_tokens']} | "
            f"{request.get('generated_tokens')} | "
            f"{request['status']} | "
            f"{request.get('error')}"
        )

async def main() -> None:
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        timeout=None,
    ) as client:
        request_tasks = [
            run_one_request(client, request_id=0, max_new_tokens=96),
            run_one_request(client, request_id=1, max_new_tokens=96),
            run_one_request(client, request_id=2, max_new_tokens=96),
            run_one_request(client, request_id=3, max_new_tokens=96),
        ]

        metrics_task = poll_metrics(client, duration_seconds=8.0)

        results = await asyncio.gather(
            asyncio.gather(*request_tasks),
            metrics_task,
        )

        request_results = results[0]

        print()
        print("request_id | max_new_tokens | ttft | latency | output_chars")
        print("--- | --- | --- | --- | ---")

        for result in request_results:
            print(
                f"{result['request_id']} | "
                f"{result['max_new_tokens']} | "
                f"{result['ttft_seconds']:.4f} | "
                f"{result['latency_seconds']:.4f} | "
                f"{result['output_chars']}"
            )

        await print_final_metrics(client)


if __name__ == "__main__":
    asyncio.run(main())