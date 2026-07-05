import pytest

from runtime.request_state import RequestState


@pytest.mark.runtime
def test_request_state_timing_metrics_are_none_until_available() -> None:
    request = RequestState(prompt="hello", max_new_tokens=4)

    assert request.queue_wait_seconds is None
    assert request.ttft_seconds is None
    assert request.decode_latency_seconds is None
    assert request.latency_seconds is None

    assert request.queue_wait_ms is None
    assert request.ttft_ms is None
    assert request.decode_latency_ms is None
    assert request.latency_ms is None


@pytest.mark.runtime
def test_request_state_sets_first_token_time_on_append_text() -> None:
    request = RequestState(prompt="hello", max_new_tokens=4)

    request.mark_admitted()
    request.mark_decoding()
    request.append_text("<tok1>")
    request.generated_tokens += 1
    request.mark_finished()

    assert request.queue_wait_seconds is not None
    assert request.ttft_seconds is not None
    assert request.decode_latency_seconds is not None
    assert request.latency_seconds is not None

    assert request.queue_wait_ms is not None
    assert request.ttft_ms is not None
    assert request.decode_latency_ms is not None
    assert request.latency_ms is not None

    assert request.generated_text == "<tok1>"


@pytest.mark.runtime
def test_request_state_lifecycle_timestamps_are_idempotent() -> None:
    request = RequestState(prompt="hello", max_new_tokens=4)

    request.mark_admitted()
    first_admit_time = request.admit_time

    request.mark_admitted()
    second_admit_time = request.admit_time

    assert first_admit_time is not None
    assert second_admit_time == first_admit_time

    request.mark_first_token()
    first_token_time = request.first_token_time

    request.mark_first_token()
    second_token_time = request.first_token_time

    assert first_token_time is not None
    assert second_token_time == first_token_time

    request.mark_finished()
    first_finish_time = request.finish_time

    request.mark_finished()
    second_finish_time = request.finish_time

    assert first_finish_time is not None
    assert second_finish_time == first_finish_time