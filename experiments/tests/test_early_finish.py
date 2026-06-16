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
        f"Request {request_id}: In LLM inference, explain transformer KV cache "
        f"in one concise paragraph."
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


async def main() -> None:
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        timeout=None,
    ) as client:
        requests = [
            run_one_request(client, request_id=0, max_new_tokens=8),
            run_one_request(client, request_id=1, max_new_tokens=16),
            run_one_request(client, request_id=2, max_new_tokens=32),
            run_one_request(client, request_id=3, max_new_tokens=32),
        ]

        results = await asyncio.gather(*requests)

    print("request_id | max_new_tokens | ttft | latency | output_chars")
    print("--- | --- | --- | --- | ---")

    for result in results:
        print(
            f"{result['request_id']} | "
            f"{result['max_new_tokens']} | "
            f"{result['ttft_seconds']:.4f} | "
            f"{result['latency_seconds']:.4f} | "
            f"{result['output_chars']}"
        )


if __name__ == "__main__":
    asyncio.run(main())