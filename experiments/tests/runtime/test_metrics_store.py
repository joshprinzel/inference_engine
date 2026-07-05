import pytest

from runtime.metrics_store import MetricsStore
from runtime.request_state import RequestState


@pytest.mark.runtime
def test_metrics_store_empty_snapshot() -> None:
    metrics = MetricsStore()

    snapshot = metrics.snapshot()

    assert snapshot["total_finished_requests"] == 0
    assert snapshot["total_generated_tokens"] == 0
    assert snapshot["requests"] == []


@pytest.mark.runtime
def test_metrics_store_records_finished_request_latency_metrics() -> None:
    metrics = MetricsStore()
    request = RequestState(prompt="hello", max_new_tokens=2)

    request.prompt_tokens = 1
    request.mark_admitted()
    request.mark_decoding()
    request.append_text("<tok1>")
    request.generated_tokens += 1
    request.append_text("<tok2>")
    request.generated_tokens += 1
    request.mark_finished()

    metrics.record_finished(request)
    snapshot = metrics.snapshot()

    assert snapshot["total_finished_requests"] == 1
    assert snapshot["total_successful_requests"] == 1
    assert snapshot["total_failed_requests"] == 0
    assert snapshot["total_generated_tokens"] == 2

    assert snapshot["avg_latency_ms"] is not None
    assert snapshot["avg_ttft_ms"] is not None
    assert snapshot["avg_queue_wait_ms"] is not None
    assert snapshot["avg_decode_latency_ms"] is not None

    recorded_request = snapshot["requests"][0]

    assert recorded_request["request_id"] == request.request_id
    assert recorded_request["status"] == "finished"
    assert recorded_request["prompt_tokens"] == 1
    assert recorded_request["generated_tokens"] == 2
    assert recorded_request["max_new_tokens"] == 2
    assert recorded_request["output_chars"] == len("<tok1><tok2>")
    assert recorded_request["error"] is None

    assert recorded_request["latency_ms"] is not None
    assert recorded_request["ttft_ms"] is not None
    assert recorded_request["queue_wait_ms"] is not None
    assert recorded_request["decode_latency_ms"] is not None


@pytest.mark.runtime
def test_metrics_store_excludes_failed_requests_from_success_token_count() -> None:
    metrics = MetricsStore()

    failed_request = RequestState(prompt="bad", max_new_tokens=4)
    failed_request.generated_tokens = 3
    failed_request.mark_failed(RuntimeError("boom"))

    metrics.record_finished(failed_request)
    snapshot = metrics.snapshot()

    assert snapshot["total_finished_requests"] == 1
    assert snapshot["total_successful_requests"] == 0
    assert snapshot["total_failed_requests"] == 1
    assert snapshot["total_generated_tokens"] == 0

    recorded_request = snapshot["requests"][0]

    assert recorded_request["status"] == "failed"
    assert recorded_request["generated_tokens"] == 3
    assert recorded_request["error"] == "RuntimeError('boom')"