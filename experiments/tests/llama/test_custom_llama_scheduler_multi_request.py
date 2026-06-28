from __future__ import annotations

import pytest

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState


pytestmark = [pytest.mark.llama, pytest.mark.slow]


def test_custom_llama_engine_runs_two_requests_through_scheduler() -> None:
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
        max_slots=2,
        step_sleep_seconds=0.0,
        idle_sleep_seconds=0.0,
    )

    request_a = RequestState(
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    request_b = RequestState(
        prompt="The capital of Germany is",
        max_new_tokens=4,
    )

    request_queue.put(request_a)
    request_queue.put(request_b)

    max_scheduler_steps = 16

    for _ in range(max_scheduler_steps):
        scheduler.step()

        if request_a.status == "finished" and request_b.status == "finished":
            break

    snapshot = scheduler.snapshot()

    print(f"request_a_text={request_a.generated_text!r}")
    print(f"request_b_text={request_b.generated_text!r}")
    print(f"request_a_tokens={request_a.generated_tokens}")
    print(f"request_b_tokens={request_b.generated_tokens}")
    print(f"request_a_prompt_tokens={request_a.prompt_tokens}")
    print(f"request_b_prompt_tokens={request_b.prompt_tokens}")
    print(f"request_a_block_table={request_a.block_table}")
    print(f"request_b_block_table={request_b.block_table}")
    print(f"scheduler_snapshot={snapshot}")

    assert request_a.status == "finished"
    assert request_b.status == "finished"

    assert request_a.generated_tokens == request_a.max_new_tokens
    assert request_b.generated_tokens == request_b.max_new_tokens

    assert request_a.generated_text
    assert request_b.generated_text

    assert request_a.generated_text.startswith("Paris")
    assert request_b.generated_text.startswith("Berlin")

    assert len(scheduler.finished) == 2
    assert scheduler.occupied_slot_count() == 0

    assert scheduler.admitted_count == 2
    assert scheduler.decode_steps == 4
    assert scheduler.tokens_generated == 8
    assert scheduler.early_finishes == 2

    assert scheduler.decode_batches_built == 4
    assert scheduler.last_decode_batch_snapshot is not None
    assert scheduler.last_decode_batch_snapshot["backend"] == "custom-llama-kv-cache-pool-gather"
    assert scheduler.last_decode_batch_snapshot["num_requests"] == 2
    assert scheduler.last_decode_batch_snapshot["uses_kv_cache"] is True
    assert scheduler.last_decode_batch_snapshot["uses_kv_cache_pool"] is True
    assert scheduler.last_decode_batch_snapshot["uses_paged_attention"] is False

    assert kv_block_manager.used_block_count() == 0
    assert kv_block_manager.free_block_count() == kv_block_manager.total_blocks