from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine_scheduler import EngineScheduler
from kv_block_manager import KVBlockManager
from metrics_store import MetricsStore
from request_queue import RequestQueue
from request_state import RequestState
from synthetic_cuda_decode_engine import SyntheticCudaDecodeEngine


def main() -> None:
    block_size_tokens = 8
    prompt_tokens = 128
    max_new_tokens = 4
    max_slots = 4

    total_blocks = max_slots * (
        (prompt_tokens + max_new_tokens + block_size_tokens - 1)
        // block_size_tokens
    )

    kv_block_manager = KVBlockManager(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
    )

    decode_engine = SyntheticCudaDecodeEngine(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
        num_layers=1,
        num_query_heads=16,
        num_kv_heads=4,
        head_dim=128,
        dtype="float16",
        device="cuda",
        attention_backend="cuda",
    )

    request_queue = RequestQueue()
    metrics_store = MetricsStore()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=max_slots,
    )

    for i in range(max_slots):
        request_queue.put(
            RequestState(
                prompt=f"synthetic prompt {i}",
                max_new_tokens=max_new_tokens,
                request_id=f"synthetic-cuda-smoke-{i}",
            )
        )

    for _ in range(20):
        scheduler.step()
        if len(scheduler.finished) == max_slots:
            break

    snapshot = scheduler.snapshot()
    print(snapshot)

    assert len(scheduler.finished) == max_slots
    assert scheduler.tokens_generated == max_slots * max_new_tokens
    assert kv_block_manager.used_block_count() == 0
    assert scheduler.decode_batches_built == max_new_tokens
    assert scheduler.last_decode_batch_snapshot is not None if hasattr(scheduler, "last_decode_batch") else True

    for request_state in scheduler.finished:
        assert request_state.status == "finished"
        assert request_state.generated_tokens == max_new_tokens
        assert request_state.generated_text == "<tok1><tok2><tok3><tok4>"

    print("passed")


if __name__ == "__main__":
    main()