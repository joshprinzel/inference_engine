from __future__ import annotations

from dataclasses import dataclass

from .request_state import RequestState


PREFILL_WORK_KIND = "prefill"
DECODE_WORK_KIND = "decode"


@dataclass(frozen=True)
class ScheduledRequestWork:
    """
    Scheduler-owned description of per-request work selected for one step.

    Today this is used for decode-only work where each selected request gets
    one scheduled token. Later this can represent prefill chunks by assigning
    multiple scheduled tokens to a request.
    """

    request_state: RequestState
    num_scheduled_tokens: int
    kind: str


def build_decode_work_plan(
    decode_batch: list[RequestState],
) -> list[ScheduledRequestWork]:
    """
    Build a per-request decode work plan for this scheduler step.

    Decode currently schedules exactly one token per request.
    """

    return [
        ScheduledRequestWork(
            request_state=request_state,
            num_scheduled_tokens=1,
            kind=DECODE_WORK_KIND,
        )
        for request_state in decode_batch
    ]


def build_prefill_work_plan(
    request_states: list[RequestState],
    max_scheduled_tokens: int | None = None,
) -> list[ScheduledRequestWork]:
    """
    Build a per-request prefill work plan.

    Today full prefill still happens when a prefill work item executes. The
    scheduled token count is now budget-aware so the scheduler can observe and test
    chunk-sized prefill decisions before engine chunk execution exists.
    """

    work_plan: list[ScheduledRequestWork] = []
    remaining_budget = max_scheduled_tokens

    for request_state in request_states:
        prefill_tokens_remaining = request_state.prefill_tokens_remaining

        if prefill_tokens_remaining <= 0:
            continue

        if remaining_budget is None:
            num_scheduled_tokens = prefill_tokens_remaining
        else:
            if remaining_budget <= 0:
                break

            num_scheduled_tokens = min(
                prefill_tokens_remaining,
                remaining_budget,
            )
            remaining_budget -= num_scheduled_tokens
        
        work_plan.append(
            ScheduledRequestWork(
                request_state=request_state,
                num_scheduled_tokens=num_scheduled_tokens,
                kind=PREFILL_WORK_KIND
            )
        )
    return work_plan


def build_candidate_work_plan(
    *,
    prefill_candidates: list[RequestState],
    decode_candidates: list[RequestState],
    max_scheduled_tokens: int | None = None
) -> list[ScheduledRequestWork]:
    """
    Build the scheduler's candidate work plan for one step.
    """

    prefill_work = build_prefill_work_plan(
        prefill_candidates,
        max_scheduled_tokens=max_scheduled_tokens
    )
    decode_work = build_decode_work_plan(decode_candidates)

    return prefill_work + decode_work


def work_of_kind(
    work_plan: list[ScheduledRequestWork],
    kind: str,
) -> list[ScheduledRequestWork]:
    """
    Return work items of a specific kind from a scheduler work plan.
    """

    return [work for work in work_plan if work.kind == kind]


def prefill_work_from_plan(
    work_plan: list[ScheduledRequestWork],
) -> list[ScheduledRequestWork]:
    """
    Return prefill work items from a scheduler work plan.
    """

    return work_of_kind(work_plan, PREFILL_WORK_KIND)


def decode_work_from_plan(
    work_plan: list[ScheduledRequestWork],
) -> list[ScheduledRequestWork]:
    """
    Return decode work items from a scheduler work plan.
    """

    return work_of_kind(work_plan, DECODE_WORK_KIND)


def requests_from_work_plan(
    work_plan: list[ScheduledRequestWork],
) -> list[RequestState]:
    """
    Extract RequestState objects from a scheduler work plan.
    """

    return [work.request_state for work in work_plan]


def total_scheduled_tokens(
    work_plan: list[ScheduledRequestWork],
) -> int:
    """
    Return the total scheduled token work in a scheduler step.
    """

    return sum(work.num_scheduled_tokens for work in work_plan)


def summarize_work_plan(
    work_plan: list[ScheduledRequestWork],
) -> dict:
    """
    Summarize a scheduler work plan for debugging and future observability.
    """

    prefill_work = prefill_work_from_plan(work_plan)
    decode_work = decode_work_from_plan(work_plan)

    return {
        "num_work_items": len(work_plan),
        "total_scheduled_tokens": total_scheduled_tokens(work_plan),
        "num_prefill_items": len(prefill_work),
        "num_decode_items": len(decode_work),
        "prefill_scheduled_tokens": total_scheduled_tokens(prefill_work),
        "decode_scheduled_tokens": total_scheduled_tokens(decode_work),
    }