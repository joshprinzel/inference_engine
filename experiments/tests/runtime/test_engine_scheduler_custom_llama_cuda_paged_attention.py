from __future__ import annotations

import pytest
import torch

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState


pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


def test_engine_scheduler_runs_custom_llama_cuda_paged_attention_single_request() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = CustomLlamaDecodeEngine(
        device="cuda",
        dtype=torch.float16,
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )

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
    )

    request_state = RequestState(
        request_id="req-0",
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    request_queue.put(request_state)

    max_steps = 16

    for _ in range(max_steps):
        scheduler.step()

        snapshot = scheduler.last_decode_batch_snapshot
        if snapshot is not None:
            assert snapshot["backend"] == "custom-llama-cuda-paged-attention-batched"
            assert snapshot["batched_decode"] is True
            assert snapshot["uses_kv_cache"] is True
            assert snapshot["uses_kv_cache_pool"] is True
            assert snapshot["uses_paged_attention"] is True
            assert snapshot["attention_backend"] == "cuda"

        if len(scheduler.finished) == 1:
            break

    assert len(scheduler.finished) == 1

    finished_request = scheduler.finished[0]

    print(f"generated_text={finished_request.generated_text!r}")
    print(f"generated_tokens={finished_request.generated_tokens}")
    print(f"scheduler_snapshot={scheduler.snapshot()}")

    assert finished_request.request_id == "req-0"
    assert finished_request.status == "finished"
    assert finished_request.generated_tokens == 4
    assert finished_request.generated_text == "Paris.\n\n"

    assert scheduler.tokens_generated == 4
    assert scheduler.decode_steps == 4
    assert scheduler.decode_batches_built == 4

    kv_snapshot = scheduler.kv_block_manager.snapshot()
    assert kv_snapshot["used_blocks"] == 0
    assert kv_snapshot["free_blocks"] == 64

    assert scheduler.last_decode_batch_snapshot is not None
    assert scheduler.last_decode_batch_snapshot["uses_paged_attention"] is True


def test_engine_scheduler_runs_custom_llama_cuda_paged_attention_two_requests() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = CustomLlamaDecodeEngine(
        device="cuda",
        dtype=torch.float16,
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )

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
    )

    france_request = RequestState(
        request_id="req-france",
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    germany_request = RequestState(
        request_id="req-germany",
        prompt="The capital of Germany is",
        max_new_tokens=4,
    )

    request_queue.put(france_request)
    request_queue.put(germany_request)

    max_steps = 16

    snapshots = []

    for _ in range(max_steps):
        scheduler.step()

        snapshot = scheduler.last_decode_batch_snapshot
        if snapshot is not None:
            snapshots.append(snapshot)

            assert snapshot["backend"] == "custom-llama-cuda-paged-attention-batched"
            assert snapshot["batched_decode"] is True
            assert snapshot["uses_kv_cache"] is True
            assert snapshot["uses_kv_cache_pool"] is True
            assert snapshot["uses_paged_attention"] is True
            assert snapshot["attention_backend"] == "cuda"

        if len(scheduler.finished) == 2:
            break

    assert len(scheduler.finished) == 2

    finished_by_id = {
        request_state.request_id: request_state
        for request_state in scheduler.finished
    }

    assert set(finished_by_id) == {"req-france", "req-germany"}

    france_finished = finished_by_id["req-france"]
    germany_finished = finished_by_id["req-germany"]

    print(f"france_text={france_finished.generated_text!r}")
    print(f"germany_text={germany_finished.generated_text!r}")
    print(f"scheduler_snapshot={scheduler.snapshot()}")

    for request in scheduler.finished:
        print(
            f"request_id={request.request_id} "
            f"status={request.status} "
            f"error={request.error!r} "
            f"generated_text={request.generated_text!r} "
            f"generated_tokens={request.generated_tokens}"
        )

    assert france_finished.status == "finished"
    assert germany_finished.status == "finished"

    assert france_finished.generated_tokens == 4
    assert germany_finished.generated_tokens == 4

    assert france_finished.generated_text == "Paris.\n\n"
    assert germany_finished.generated_text == "Berlin.\n\n"

    assert scheduler.tokens_generated == 8
    assert scheduler.decode_steps == 4
    assert scheduler.decode_batches_built == 4

    kv_snapshot = scheduler.kv_block_manager.snapshot()

    assert kv_snapshot["used_blocks"] == 0
    assert kv_snapshot["free_blocks"] == 64
    assert kv_snapshot["active_requests"] == 0
    assert kv_snapshot["block_tables"] == {}

    assert len(snapshots) >= 1
    assert any(snapshot["num_requests"] == 2 for snapshot in snapshots)
    assert scheduler.last_decode_batch_snapshot is not None
    assert scheduler.last_decode_batch_snapshot["uses_paged_attention"] is True


def test_engine_scheduler_custom_llama_cuda_paged_attention_multiblock_request() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    block_size_tokens = 4

    engine = CustomLlamaDecodeEngine(
        device="cuda",
        dtype=torch.float16,
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=block_size_tokens,
    )

    request_queue = RequestQueue()
    metrics_store = MetricsStore()
    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=block_size_tokens,
    )

    scheduler = EngineScheduler(
        decode_engine=engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
    )

    request_state = RequestState(
        request_id="req-france-multiblock",
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    prompt_tokens = engine.count_prompt_tokens(request_state.prompt)
    total_tokens = prompt_tokens + request_state.max_new_tokens

    expected_blocks = (total_tokens + block_size_tokens - 1) // block_size_tokens

    assert prompt_tokens == 6
    assert total_tokens == 10
    assert expected_blocks == 3

    request_queue.put(request_state)

    max_steps = 16
    snapshots = []
    active_block_counts = []

    for _ in range(max_steps):
        scheduler.step()

        snapshot = scheduler.snapshot()
        snapshots.append(snapshot)

        kv_snapshot = snapshot["kv_cache"]
        active_block_counts.append(kv_snapshot["used_blocks"])

        decode_snapshot = scheduler.last_decode_batch_snapshot
        if decode_snapshot is not None:
            assert decode_snapshot["backend"] == "custom-llama-cuda-paged-attention-batched"
            assert decode_snapshot["batched_decode"] is True
            assert decode_snapshot["uses_kv_cache"] is True
            assert decode_snapshot["uses_kv_cache_pool"] is True
            assert decode_snapshot["uses_paged_attention"] is True
            assert decode_snapshot["attention_backend"] == "cuda"

        if len(scheduler.finished) == 1:
            break

    assert len(scheduler.finished) == 1

    finished_request = scheduler.finished[0]

    print(f"generated_text={finished_request.generated_text!r}")
    print(f"generated_tokens={finished_request.generated_tokens}")
    print(f"active_block_counts={active_block_counts}")
    print(f"scheduler_snapshot={scheduler.snapshot()}")

    assert finished_request.request_id == "req-france-multiblock"
    assert finished_request.status == "finished"
    assert finished_request.generated_tokens == 4
    assert finished_request.generated_text == "Paris.\n\n"

    assert scheduler.tokens_generated == 4
    assert scheduler.decode_steps == 4
    assert scheduler.decode_batches_built == 4

    # During execution, the request should have occupied 3 physical KV blocks:
    # prompt_len=6, max_new_tokens=4, block_size=4 -> ceil(10 / 4) = 3.
    assert max(active_block_counts) == expected_blocks

    final_kv_snapshot = scheduler.kv_block_manager.snapshot()

    assert final_kv_snapshot["used_blocks"] == 0
    assert final_kv_snapshot["free_blocks"] == 64
    assert final_kv_snapshot["active_requests"] == 0
    assert final_kv_snapshot["block_tables"] == {}

    assert scheduler.last_decode_batch_snapshot is not None
    assert scheduler.last_decode_batch_snapshot["uses_paged_attention"] is True