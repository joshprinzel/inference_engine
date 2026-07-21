import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    decode_engine = CustomLlamaDecodeEngine(
        attention_backend_name="cuda",
        total_kv_blocks=128,
        block_size_tokens=16,
    )
    kv_block_manager = KVBlockManager(
        total_blocks=128,
        block_size_tokens=16,
    )
    request_queue = RequestQueue()
    metrics_store = MetricsStore()

    scheduler = EngineScheduler(
        decode_engine=decode_engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=1,
        max_scheduled_tokens_per_step=4,
    )

    request = RequestState(
        prompt=(
            "Explain why chunked prefill matters for LLM inference serving "
            "in one concise paragraph."
        ),
        max_new_tokens=8,
        request_id="req-1",
    )

    request_queue.put(request)

    for step_index in range(32):
        scheduler.step()
        snapshot = scheduler.snapshot()

        slot = snapshot["slots"][0]
        print(
            {
                "step": step_index + 1,
                "status": None if slot is None else slot["status"],
                "prompt_tokens": None if slot is None else slot["prompt_tokens"],
                "num_computed_tokens": (
                    None if slot is None else slot["num_computed_tokens"]
                ),
                "prefill_remaining": (
                    None if slot is None else slot["prefill_tokens_remaining"]
                ),
                "generated_tokens": (
                    None if slot is None else slot["generated_tokens"]
                ),
                "text": request.generated_text,
                "candidate_work": snapshot["last_candidate_work_plan_summary"],
                "executed_work": snapshot["last_executed_work_plan_summary"],
            }
        )

        if request.status == "finished":
            break

    print("\nFinal request state:")
    print(f"status={request.status}")
    print(f"prompt_tokens={request.prompt_tokens}")
    print(f"num_computed_tokens={request.num_computed_tokens}")
    print(f"generated_tokens={request.generated_tokens}")
    print(f"generated_text={request.generated_text!r}")

    assert request.generated_tokens > 0
    assert request.num_computed_tokens == (
        request.prompt_tokens + request.generated_tokens
    )
    assert request.status in {"decoding", "finished"}


if __name__ == "__main__":
    main()