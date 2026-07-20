import pytest

from runtime.request_state import RequestState
from runtime.scheduler_work_plan import (
    DECODE_WORK_KIND,
    PREFILL_WORK_KIND,
    build_candidate_work_plan,
    build_decode_work_plan,
    build_prefill_work_plan,
    decode_work_from_plan,
    prefill_work_from_plan,
    summarize_work_plan,
    total_scheduled_tokens,
)


def make_request(
    *,
    request_id: str,
    prompt: str = "hello",
    max_new_tokens: int = 4,
    prompt_tokens: int = 0,
    num_computed_tokens: int = 0,
    generated_tokens: int = 0,
) -> RequestState:
    request = RequestState(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )
    request.prompt_tokens = prompt_tokens
    request.num_computed_tokens = num_computed_tokens
    request.generated_tokens = generated_tokens
    return request


@pytest.mark.runtime
def test_build_decode_work_plan_assigns_one_decode_token_per_request() -> None:
    request_a = make_request(request_id="req-a")
    request_b = make_request(request_id="req-b")

    work_plan = build_decode_work_plan([request_a, request_b])

    assert [work.request_state for work in work_plan] == [request_a, request_b]
    assert [work.num_scheduled_tokens for work in work_plan] == [1, 1]
    assert [work.kind for work in work_plan] == [
        DECODE_WORK_KIND,
        DECODE_WORK_KIND,
    ]


@pytest.mark.runtime
def test_total_scheduled_tokens_sums_work_plan_tokens() -> None:
    request_a = make_request(request_id="req-a")
    request_b = make_request(request_id="req-b")

    work_plan = build_decode_work_plan([request_a, request_b])

    assert total_scheduled_tokens(work_plan) == 2


@pytest.mark.runtime
def test_build_prefill_work_plan_uses_prefill_tokens_remaining() -> None:
    request = make_request(
        request_id="req-1",
        prompt="hello world",
        prompt_tokens=10,
        num_computed_tokens=4,
    )

    work_plan = build_prefill_work_plan([request])

    assert len(work_plan) == 1
    assert work_plan[0].request_state is request
    assert work_plan[0].num_scheduled_tokens == 6
    assert work_plan[0].kind == PREFILL_WORK_KIND


@pytest.mark.runtime
def test_build_prefill_work_plan_skips_requests_with_no_prefill_remaining() -> None:
    request = make_request(
        request_id="req-1",
        prompt="hello world",
        prompt_tokens=10,
        num_computed_tokens=10,
    )

    work_plan = build_prefill_work_plan([request])

    assert work_plan == []


@pytest.mark.runtime
def test_build_candidate_work_plan_combines_prefill_and_decode_work() -> None:
    prefill_request = make_request(
        request_id="prefill-req",
        prompt="prefill",
        prompt_tokens=10,
        num_computed_tokens=4,
    )
    decode_request = make_request(
        request_id="decode-req",
        prompt="decode",
        prompt_tokens=3,
        num_computed_tokens=3,
        generated_tokens=0,
    )

    work_plan = build_candidate_work_plan(
        prefill_candidates=[prefill_request],
        decode_candidates=[decode_request],
    )

    assert [work.request_state.request_id for work in work_plan] == [
        "prefill-req",
        "decode-req",
    ]
    assert [work.kind for work in work_plan] == [
        PREFILL_WORK_KIND,
        DECODE_WORK_KIND,
    ]
    assert [work.num_scheduled_tokens for work in work_plan] == [6, 1]
    assert total_scheduled_tokens(work_plan) == 7


@pytest.mark.runtime
def test_work_plan_filters_by_kind() -> None:
    prefill_request = make_request(
        request_id="prefill-req",
        prompt="prefill",
        prompt_tokens=10,
        num_computed_tokens=4,
    )
    decode_request = make_request(
        request_id="decode-req",
        prompt="decode",
    )

    work_plan = build_candidate_work_plan(
        prefill_candidates=[prefill_request],
        decode_candidates=[decode_request],
    )

    prefill_work = prefill_work_from_plan(work_plan)
    decode_work = decode_work_from_plan(work_plan)

    assert [work.request_state.request_id for work in prefill_work] == [
        "prefill-req"
    ]
    assert [work.request_state.request_id for work in decode_work] == [
        "decode-req"
    ]
    assert [work.kind for work in prefill_work] == [PREFILL_WORK_KIND]
    assert [work.kind for work in decode_work] == [DECODE_WORK_KIND]


@pytest.mark.runtime
def test_summarize_work_plan_reports_counts_and_token_totals() -> None:
    prefill_request = make_request(
        request_id="prefill-req",
        prompt="prefill",
        prompt_tokens=10,
        num_computed_tokens=4,
    )
    decode_request = make_request(
        request_id="decode-req",
        prompt="decode",
    )

    work_plan = build_candidate_work_plan(
        prefill_candidates=[prefill_request],
        decode_candidates=[decode_request],
    )

    assert summarize_work_plan(work_plan) == {
        "num_work_items": 2,
        "total_scheduled_tokens": 7,
        "num_prefill_items": 1,
        "num_decode_items": 1,
        "prefill_scheduled_tokens": 6,
        "decode_scheduled_tokens": 1,
    }

@pytest.mark.runtime
def test_build_prefill_work_plan_applies_token_budget_to_single_request() -> None:
    request = make_request(
        request_id="req-1",
        prompt="hello world",
        prompt_tokens=10,
        num_computed_tokens=4,
    )

    work_plan = build_prefill_work_plan(
        [request],
        max_scheduled_tokens=3,
    )

    assert len(work_plan) == 1
    assert work_plan[0].request_state is request
    assert work_plan[0].num_scheduled_tokens == 3
    assert work_plan[0].kind == PREFILL_WORK_KIND


@pytest.mark.runtime
def test_build_prefill_work_plan_applies_token_budget_across_requests() -> None:
    request_a = make_request(
        request_id="req-a",
        prompt_tokens=10,
        num_computed_tokens=4,
    )
    request_b = make_request(
        request_id="req-b",
        prompt_tokens=8,
        num_computed_tokens=2,
    )

    work_plan = build_prefill_work_plan(
        [request_a, request_b],
        max_scheduled_tokens=8,
    )

    assert [work.request_state.request_id for work in work_plan] == [
        "req-a",
        "req-b",
    ]
    assert [work.num_scheduled_tokens for work in work_plan] == [6, 2]
    assert total_scheduled_tokens(work_plan) == 8


@pytest.mark.runtime
def test_build_prefill_work_plan_returns_empty_when_budget_is_zero() -> None:
    request = make_request(
        request_id="req-1",
        prompt_tokens=10,
        num_computed_tokens=4,
    )

    work_plan = build_prefill_work_plan(
        [request],
        max_scheduled_tokens=0,
    )

    assert work_plan == []