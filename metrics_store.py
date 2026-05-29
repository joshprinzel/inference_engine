from threading import Lock
from typing import Any

from request_state import RequestState

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
                    "latency_seconds": request_state.latency_seconds,
                    "max_new_tokens": request_state.max_new_tokens,
                    "output_chars": len(request_state.generated_text),
                    "error": request_state.error,
                }
            )
    
    def snapshot(self) -> dict[str,Any]:
        with self._lock:
            requests = list(self.finished_requests)

        if not requests:
            return {
                "total_finished_requests": 0,
                "requests": []
            }
        
        latencies = [r["latency_seconds"] for r in requests]
        ttfts = [r["ttft_seconds"] for r in requests]
        queue_waits = [r["queue_wait_seconds"] for r in requests]

        return {
            "total_finished_requests": len(requests),
            "avg_latency_seconds": sum(latencies) / len(latencies),
            "avg_ttft_seconds": sum(ttfts) / len(ttfts),
            "avg_queue_wait_seconds": sum(queue_waits) / len(queue_waits),
            "requests": requests[-50:],
        }
    
