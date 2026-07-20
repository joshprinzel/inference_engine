import pytest

from runtime.request_state import RequestState


@pytest.mark.runtime
def test_request_state_prefill_decode_remaining_estimates() -> None:
    request = RequestState(prompt="hello", max_new_tokens=8)

    request.prompt_tokens = 10
    request.num_computed_tokens = 4
    request.generated_tokens = 3

    assert request.prefill_tokens_total == 10
    assert request.prefill_tokens_remaining == 6
    assert request.decode_tokens_total == 8
    assert request.decode_tokens_remaining == 5
    assert request.estimated_total_tokens_remaining == 11


@pytest.mark.runtime
def test_request_state_remaining_estimates_clamp_at_zero() -> None:
    request = RequestState(prompt="hello", max_new_tokens=8)

    request.prompt_tokens = 10
    request.num_computed_tokens = 12
    request.generated_tokens = 10

    assert request.prefill_tokens_remaining == 0
    assert request.decode_tokens_remaining == 0
    assert request.estimated_total_tokens_remaining == 0