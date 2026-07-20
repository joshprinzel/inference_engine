import pytest

from runtime.decode_engine import DecodeStepOutput, RequestDecodeOutput
from runtime.engine_scheduler import EngineScheduler
from runtime.request_state import RequestState


DECODE_WORK_SUMMARY_ONE_REQUEST = {
    "num_work_items": 1,
    "total_scheduled_tokens": 1,
    "num_prefill_items": 0,
    "num_decode_items": 1,
    "prefill_scheduled_tokens": 0,
    "decode_scheduled_tokens": 1,
}


class FakeDecodeEngine:
    @property
    def device(self) -> str:
        return "cpu"

    def __init__(self) -> None:
        self.decode_calls: list[list[str]] = []

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    def prefill_request(self, request_state: RequestState) -> None:
        request_state.prompt_tokens = self.count_prompt_tokens(
            request_state.prompt
        )
        request_state.num_computed_tokens = request_state.prompt_tokens
        request_state.next_token = 1

    def init_request_state(self, request_state: RequestState) -> None:
        self.prefill_request(request_state)

    def decode_step(
        self,
        request_states: list[RequestState],
        kv_block_manager,
    ) -> DecodeStepOutput:
        self.decode_calls.append(
            [
                str(request_state.request_id)
                for request_state in request_states
            ]
        )

        request_outputs: list[RequestDecodeOutput] = []

        for request_state in request_states:
            request_state.generated_tokens += 1

            request_outputs.append(
                RequestDecodeOutput(
                    request_id=request_state.request_id,
                    text="A",
                    generated_tokens=1,
                    finished=False,
                )
            )

        return DecodeStepOutput(
            request_outputs=request_outputs,
            backend_ms=1.25,
            decode_batch_snapshot={
                "backend": "fake",
                "num_requests": len(request_states),
            },
        )


class FakeMetricsStore:
    def __init__(self) -> None:
        self.recorded: list[RequestState] = []

    def record_finished(self, request_state: RequestState) -> None:
        self.recorded.append(request_state)


class FakeKVBlockManager:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str, int]] = []
        self.freed: list[str] = []

    def ensure_capacity_for_token(
        self,
        *,
        request_id: str,
        token_position: int,
    ) -> None:
        self.ensure_calls.append((request_id, token_position))

    def get_block_tables(self, request_id: str) -> list[int]:
        return [0]

    def free(self, request_id: str) -> None:
        self.freed.append(request_id)

    def snapshot(self) -> dict:
        return {
            "fake": True,
            "used_blocks": 1,
            "free_blocks": 7,
        }

    def used_block_count(self) -> int:
        return 1

    def free_block_count(self) -> int:
        return 7

    def utilization(self) -> float:
        return 1 / 8


class EmptyDecodeBatchPolicy:
    name = "empty_decode_batch"

    def select_admissions(
        self,
        *,
        waiting,
        active,
        available_slots,
        kv_block_manager,
        decode_engine,
    ):
        return []

    def select_decode_batch(
        self,
        *,
        active,
        kv_block_manager,
        max_batch_size=None,
    ):
        return []


def make_scheduler(
    *,
    max_slots: int = 1,
    scheduling_policy=None,
) -> tuple[
    EngineScheduler,
    FakeDecodeEngine,
    FakeMetricsStore,
    FakeKVBlockManager,
]:
    decode_engine = FakeDecodeEngine()
    metrics_store = FakeMetricsStore()
    kv_block_manager = FakeKVBlockManager()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=None,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=max_slots,
        scheduling_policy=scheduling_policy,
    )

    return scheduler, decode_engine, metrics_store, kv_block_manager


def make_decode_ready_request(
    *,
    request_id: str = "req-1",
    prompt: str = "hello world",
    prompt_tokens: int = 2,
    generated_tokens: int = 0,
    max_new_tokens: int = 4,
) -> RequestState:
    request = RequestState(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )
    request.status = "decoding"
    request.prompt_tokens = prompt_tokens
    request.generated_tokens = generated_tokens
    request.num_computed_tokens = prompt_tokens + generated_tokens
    request.block_table = [0]
    request.next_token = 1
    return request


@pytest.mark.runtime
def test_decode_active_requests_runs_decode_and_applies_output() -> None:
    scheduler, decode_engine, metrics_store, kv_block_manager = make_scheduler()

    request = make_decode_ready_request()
    scheduler.slots[0] = request

    scheduler.decode_active_requests()

    assert scheduler.decode_steps == 1
    assert scheduler.decode_batches_built == 1
    assert scheduler.tokens_generated == 1
    assert scheduler.last_backend_ms == 1.25
    assert scheduler.last_decode_batch_snapshot == {
        "backend": "fake",
        "num_requests": 1,
    }
    assert (
        scheduler.last_candidate_work_plan_summary
        == DECODE_WORK_SUMMARY_ONE_REQUEST
    )
    assert (
        scheduler.last_executed_work_plan_summary
        == DECODE_WORK_SUMMARY_ONE_REQUEST
    )

    assert decode_engine.decode_calls == [["req-1"]]
    assert kv_block_manager.ensure_calls == [("req-1", 2)]

    assert request.generated_text == "A"
    assert request.num_computed_tokens == 3
    assert request.status == "decoding"
    assert scheduler.slots[0] is request
    assert metrics_store.recorded == []
    assert kv_block_manager.freed == []

    snapshot = scheduler.snapshot()
    assert (
        snapshot["last_candidate_work_plan_summary"]
        == DECODE_WORK_SUMMARY_ONE_REQUEST
    )
    assert (
        snapshot["last_executed_work_plan_summary"]
        == DECODE_WORK_SUMMARY_ONE_REQUEST
    )


@pytest.mark.runtime
def test_decode_active_requests_empty_policy_batch_stalls_without_engine_call() -> None:
    (
        scheduler,
        decode_engine,
        metrics_store,
        kv_block_manager,
    ) = make_scheduler(
        scheduling_policy=EmptyDecodeBatchPolicy(),
    )

    request = make_decode_ready_request()
    scheduler.slots[0] = request

    scheduler.decode_active_requests()

    assert scheduler.decode_steps == 1
    assert scheduler.decode_stalls == 1
    assert decode_engine.decode_calls == []
    assert scheduler.tokens_generated == 0
    assert scheduler.decode_batches_built == 0
    assert scheduler.last_candidate_work_plan_summary is None
    assert scheduler.last_executed_work_plan_summary is None

    assert request.generated_text == ""
    assert scheduler.slots[0] is request
    assert metrics_store.recorded == []
    assert kv_block_manager.freed == []


@pytest.mark.runtime
def test_mark_request_prefill_complete_rejects_incomplete_prefill() -> None:
    scheduler = EngineScheduler(
        decode_engine=FakeDecodeEngine(),
        request_queue=None,
        metrics_store=FakeMetricsStore(),
        kv_block_manager=FakeKVBlockManager(),
        max_slots=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request.status = "prefill"
    request.prompt_tokens = 10
    request.num_computed_tokens = 4

    with pytest.raises(RuntimeError, match="prefill tokens remain"):
        scheduler.mark_request_prefill_complete(request)


@pytest.mark.runtime
def test_mark_request_prefill_complete_sets_decoding_status() -> None:
    scheduler = EngineScheduler(
        decode_engine=FakeDecodeEngine(),
        request_queue=None,
        metrics_store=FakeMetricsStore(),
        kv_block_manager=FakeKVBlockManager(),
        max_slots=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request.status = "prefill"
    request.prompt_tokens = 10
    request.num_computed_tokens = 10

    scheduler.mark_request_prefill_complete(request)

    assert request.status == "decoding"

@pytest.mark.runtime
def test_prefill_active_request_states_returns_prefill_incomplete_requests() -> None:
    scheduler, _, _, _ = make_scheduler(max_slots=2)

    prefill_request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="prefill-req",
    )
    prefill_request.status = "prefill"
    prefill_request.prompt_tokens = 10
    prefill_request.num_computed_tokens = 4

    decode_request = make_decode_ready_request(request_id="decode-req")

    scheduler.slots[0] = prefill_request
    scheduler.slots[1] = decode_request

    assert scheduler.prefill_active_request_states() == [prefill_request]


@pytest.mark.runtime
def test_decode_active_requests_skips_prefill_incomplete_request() -> None:
    scheduler, decode_engine, metrics_store, kv_block_manager = make_scheduler()

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request.status = "prefill"
    request.prompt_tokens = 10
    request.num_computed_tokens = 4
    request.generated_tokens = 0
    request.block_table = [0]
    request.next_token = 1

    scheduler.slots[0] = request

    scheduler.decode_active_requests()

    assert scheduler.decode_steps == 0
    assert decode_engine.decode_calls == []
    assert kv_block_manager.ensure_calls == []
    assert scheduler.tokens_generated == 0
    assert request.generated_text == ""
    assert scheduler.slots[0] is request
    assert metrics_store.recorded == []
    assert kv_block_manager.freed == []