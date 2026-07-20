import pytest
from queue import Empty

from runtime.decode_engine import DecodeStepOutput, RequestDecodeOutput
from runtime.engine_scheduler import EngineScheduler
from runtime.request_state import RequestState


class FakeDecodeEngine:
    @property
    def device(self) -> str:
        return "cpu"

    def __init__(self) -> None:
        self.prefilled_requests: list[str] = []
        self.decode_calls: list[list[str]] = []

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    def prefill_request(self, request_state: RequestState) -> None:
        self.prefilled_requests.append(str(request_state.request_id))

        request_state.input_ids = [1] * request_state.prompt_tokens
        request_state.next_token = 1
        request_state.num_computed_tokens = request_state.prompt_tokens
    
    def prefill_chunk(
            self,
            request_state: RequestState,
            num_tokens: int,
            kv_block_manager
    ) -> None:
        self.prefilled_requests.append(str(request_state.request_id))

        if request_state.input_ids is None:
            request_state.input_ids = [1] * request_state.prompt_tokens

        request_state.num_computed_tokens += num_tokens

        if request_state.num_computed_tokens > request_state.prompt_tokens:
            request_state.num_computed_tokens = request_state.prompt_tokens
        
        if request_state.prefill_tokens_remaining == 0:
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
        self.allocated: dict[str, list[int]] = {}
        self.freed: list[str] = []
        self.ensure_calls: list[tuple[str, int]] = []

    def can_allocate_tokens(self, num_tokens: int) -> bool:
        return True

    def allocate_for_tokens(
        self,
        request_id: str,
        num_tokens: int,
    ) -> list[int]:
        self.allocated[request_id] = [0]
        return [0]

    def free(self, request_id: str) -> None:
        self.freed.append(request_id)

    def ensure_capacity_for_token(
        self,
        *,
        request_id: str,
        token_position: int,
    ) -> None:
        self.ensure_calls.append((request_id, token_position))

    def get_block_tables(self, request_id: str) -> list[int]:
        return [0]


class FakeRequestQueue:
    def __init__(self) -> None:
        self.items: list[RequestState] = []

    def get_nowait(self) -> RequestState:
        if not self.items:
            raise Empty
        return self.items.pop(0)


@pytest.mark.runtime
def test_admit_waiting_requests_reserves_kv_and_places_prefill_request() -> None:
    decode_engine = FakeDecodeEngine()
    metrics_store = FakeMetricsStore()
    kv_block_manager = FakeKVBlockManager()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=None,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    scheduler.waiting.append(request)

    scheduler.admit_waiting_requests()

    assert scheduler.waiting == []
    assert scheduler.slots[0] is request
    assert scheduler.admitted_count == 1

    assert request.status == "prefill"
    assert request.prompt_tokens == 2
    assert request.block_table == [0]
    assert request.num_computed_tokens == 0
    assert request.prefill_tokens_remaining == 2
    assert decode_engine.prefilled_requests == []

    assert kv_block_manager.allocated == {"req-1": [0]}
    assert kv_block_manager.freed == []
    assert metrics_store.recorded == []


@pytest.mark.runtime
def test_step_admits_prefills_and_decodes_in_one_scheduler_step() -> None:
    decode_engine = FakeDecodeEngine()
    metrics_store = FakeMetricsStore()
    kv_block_manager = FakeKVBlockManager()
    request_queue = FakeRequestQueue()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request_queue.items.append(request)

    scheduler.step()

    assert request.status == "decoding"
    assert request.prompt_tokens == 2
    assert request.num_computed_tokens == 3
    assert request.generated_tokens == 1
    assert request.generated_text == "A"

    assert decode_engine.prefilled_requests == ["req-1"]
    assert decode_engine.decode_calls == [["req-1"]]
    assert kv_block_manager.ensure_calls == [("req-1", 2)]

    assert scheduler.tokens_generated == 1
    assert scheduler.engine_step == 1
    assert metrics_store.recorded == []
    assert kv_block_manager.freed == []


@pytest.mark.runtime
def test_run_prefill_for_active_requests_records_prefill_work_summary() -> None:
    decode_engine = FakeDecodeEngine()
    metrics_store = FakeMetricsStore()
    kv_block_manager = FakeKVBlockManager()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=None,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request.status = "prefill"
    request.prompt_tokens = 2
    request.num_computed_tokens = 0
    request.block_table = [0]

    scheduler.slots[0] = request

    scheduler.run_prefill_for_active_requests()

    expected_summary = {
        "num_work_items": 1,
        "total_scheduled_tokens": 2,
        "num_prefill_items": 1,
        "num_decode_items": 0,
        "prefill_scheduled_tokens": 2,
        "decode_scheduled_tokens": 0,
    }

    assert scheduler.last_candidate_work_plan_summary == expected_summary
    assert scheduler.last_executed_work_plan_summary == expected_summary
    assert request.status == "decoding"
    assert request.num_computed_tokens == 2

@pytest.mark.runtime
def test_run_prefill_for_active_requests_uses_prefill_token_budget_in_summary() -> None:
    decode_engine = FakeDecodeEngine()
    metrics_store = FakeMetricsStore()
    kv_block_manager = FakeKVBlockManager()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=None,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
        max_scheduled_tokens_per_step=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request.status = "prefill"
    request.prompt_tokens = 2
    request.num_computed_tokens = 0
    request.block_table = [0]

    scheduler.slots[0] = request

    scheduler.run_prefill_for_active_requests()

    expected_summary = {
        "num_work_items": 1,
        "total_scheduled_tokens": 1,
        "num_prefill_items": 1,
        "num_decode_items": 0,
        "prefill_scheduled_tokens": 1,
        "decode_scheduled_tokens": 0,
    }

    assert scheduler.last_candidate_work_plan_summary == expected_summary
    assert scheduler.last_executed_work_plan_summary == expected_summary

    
    assert request.status == "prefill"
    assert request.num_computed_tokens == 1
    assert request.prefill_tokens_remaining == 1
    assert decode_engine.decode_calls == []

@pytest.mark.runtime
def test_step_chunks_prefill_across_steps_before_decode() -> None:
    decode_engine = FakeDecodeEngine()
    metrics_store = FakeMetricsStore()
    kv_block_manager = FakeKVBlockManager()
    request_queue = FakeRequestQueue()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
        max_scheduled_tokens_per_step=1,
    )

    request = RequestState(
        prompt="hello world",
        max_new_tokens=4,
        request_id="req-1",
    )
    request_queue.items.append(request)

    scheduler.step()

    assert request.status == "prefill"
    assert request.prompt_tokens == 2
    assert request.num_computed_tokens == 1
    assert request.prefill_tokens_remaining == 1
    assert request.generated_tokens == 0
    assert request.generated_text == ""
    assert decode_engine.decode_calls == []
    assert scheduler.engine_step == 1

    scheduler.step()

    assert request.status == "decoding"
    assert request.prompt_tokens == 2
    assert request.num_computed_tokens == 3
    assert request.prefill_tokens_remaining == 0
    assert request.generated_tokens == 1
    assert request.generated_text == "A"
    assert decode_engine.decode_calls == [["req-1"]]
    assert scheduler.engine_step == 2