from __future__ import annotations

import pytest

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState


pytestmark = [pytest.mark.llama, pytest.mark.slow]


def test_custom_llama_engine_runs_through_scheduler() -> None:
    engine = CustomLlamaDecodeEngine()

    request_queue = RequestQueue()
    metrics_store = MetricsStore()
    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    scheduler = EngineScheduler(
        decode_engine=engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
        step_sleep_seconds=0.0,
        idle_sleep_seconds=0.0,
    )

    request_state = RequestState(
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    request_queue.put(request_state)

    max_scheduler_steps = 16

    for _ in range(max_scheduler_steps):
        scheduler.step()

        if request_state.status == "finished":
            break

    snapshot = scheduler.snapshot()

    print(f"generated_text={request_state.generated_text!r}")
    print(f"generated_tokens={request_state.generated_tokens}")
    print(f"status={request_state.status}")
    print(f"prompt_tokens={request_state.prompt_tokens}")
    print(f"num_computed_tokens={request_state.num_computed_tokens}")
    print(f"block_table={request_state.block_table}")
    print(f"scheduler_snapshot={snapshot}")

    assert request_state.status == "finished"
    assert request_state.generated_text.startswith("Paris")
    assert request_state.generated_tokens == request_state.max_new_tokens
    assert request_state.prompt_tokens == 6
    assert request_state.num_computed_tokens == (
        request_state.prompt_tokens + request_state.generated_tokens
    )

    assert len(scheduler.finished) == 1
    assert scheduler.finished[0] is request_state
    assert scheduler.occupied_slot_count() == 0

    assert scheduler.admitted_count == 1
    assert scheduler.decode_steps == request_state.max_new_tokens
    assert scheduler.tokens_generated == request_state.max_new_tokens
    assert request_state.generated_tokens == request_state.max_new_tokens
    assert scheduler.early_finishes == 1

    assert scheduler.last_backend_ms is not None
    assert scheduler.last_decode_batch_snapshot is not None
    assert scheduler.last_decode_batch_snapshot["backend"] == "custom-llama-kv-cache-pool-gather"
    assert scheduler.last_decode_batch_snapshot["uses_kv_cache"] is True
    assert scheduler.last_decode_batch_snapshot["uses_kv_cache_pool"] is True
    assert scheduler.last_decode_batch_snapshot["uses_paged_attention"] is False

    assert kv_block_manager.used_block_count() == 0
    assert kv_block_manager.free_block_count() == kv_block_manager.total_blocks