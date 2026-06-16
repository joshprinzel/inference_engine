from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine_scheduler import EngineScheduler
from hf_decode_engine import HFDecodeEngine
from kv_block_manager import KVBlockManager
from metrics_store import MetricsStore
from model_runner import ModelRunner
from request_queue import RequestQueue
from request_state import RequestState


def main() -> None:
    runner = ModelRunner()
    decode_engine = HFDecodeEngine(runner)

    request_queue = RequestQueue()
    metrics_store = MetricsStore()
    kv_block_manager = KVBlockManager(
        total_blocks=256,
        block_size_tokens=8,
    )

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=2,
    )

    request = RequestState(
        prompt="Write one short sentence about GPUs.",
        max_new_tokens=2,
        request_id="engine-scheduler-smoke-0",
    )

    request_queue.put(request)

    for _ in range(20):
        scheduler.step()
        if len(scheduler.finished) == 1:
            break

    snapshot = scheduler.snapshot()
    print(snapshot)

    assert len(scheduler.finished) == 1
    finished = scheduler.finished[0]

    assert finished.status == "finished"
    assert finished.generated_tokens == 2
    assert scheduler.tokens_generated == 2
    assert kv_block_manager.used_block_count() == 0

    print("generated_text:", finished.generated_text)
    print("passed")


if __name__ == "__main__":
    main()