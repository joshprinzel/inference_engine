from threading import Lock
from typing import Any

from .request_state import RequestState


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.finished_requests: list[dict[str, Any]] = []

    def record_finished(self, request_state: RequestState) -> None:
        with self._lock:
            self.finished_requests.append(
                {
                    "request_id": request_state.request_id,
                    "status": request_state.status,
                    "queue_wait_seconds": request_state.queue_wait_seconds,
                    "ttft_seconds": request_state.ttft_seconds,
                    "decode_latency_seconds": request_state.decode_latency_seconds,
                    "latency_seconds": request_state.latency_seconds,
                    "queue_wait_ms": request_state.queue_wait_ms,
                    "ttft_ms": request_state.ttft_ms,
                    "decode_latency_ms": request_state.decode_latency_ms,
                    "latency_ms": request_state.latency_ms,
                    "max_new_tokens": request_state.max_new_tokens,
                    "generated_tokens": request_state.generated_tokens,
                    "prompt_tokens": request_state.prompt_tokens,
                    "output_chars": len(request_state.generated_text),
                    "error": request_state.error,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = list(self.finished_requests)

        if not requests:
            return {
                "total_finished_requests": 0,
                "total_generated_tokens": 0,
                "requests": [],
            }

        successful_requests = [
            request
            for request in requests
            if request.get("error") is None
        ]

        generated_tokens = [
            request.get("generated_tokens", 0)
            for request in successful_requests
        ]

        latency_ms_values = [
            request["latency_ms"]
            for request in successful_requests
            if request.get("latency_ms") is not None
        ]

        ttft_ms_values = [
            request["ttft_ms"]
            for request in successful_requests
            if request.get("ttft_ms") is not None
        ]

        queue_wait_ms_values = [
            request["queue_wait_ms"]
            for request in successful_requests
            if request.get("queue_wait_ms") is not None
        ]

        decode_latency_ms_values = [
            request["decode_latency_ms"]
            for request in successful_requests
            if request.get("decode_latency_ms") is not None
        ]

        return {
            "total_finished_requests": len(requests),
            "total_successful_requests": len(successful_requests),
            "total_failed_requests": len(requests) - len(successful_requests),
            "total_generated_tokens": sum(generated_tokens),
            "avg_latency_ms": _mean(latency_ms_values),
            "avg_ttft_ms": _mean(ttft_ms_values),
            "avg_queue_wait_ms": _mean(queue_wait_ms_values),
            "avg_decode_latency_ms": _mean(decode_latency_ms_values),
            "requests": requests[-50:],
        }