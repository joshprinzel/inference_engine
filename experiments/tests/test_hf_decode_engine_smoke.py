from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_decode_engine import HFDecodeEngine
from kv_block_manager import KVBlockManager
from model_runner import ModelRunner
from request_state import RequestState


def main() -> None:
    runner = ModelRunner()
    engine = HFDecodeEngine(runner)

    request = RequestState(
        prompt="Write one short sentence about GPUs.",
        max_new_tokens=2,
        request_id="smoke-0",
    )

    engine.init_request_state(request)

    assert request.status == "decoding"
    assert request.next_token is not None
    assert request.prompt_tokens > 0

    kv_block_manager = KVBlockManager(
        total_blocks=128,
        block_size_tokens=8,
    )

    output = engine.decode_step(
        request_states=[request],
        kv_block_manager=kv_block_manager,
    )

    print(output)

    assert len(output.request_outputs) == 1
    assert request.generated_tokens == 1
    assert request.next_token is not None

    print("passed")


if __name__ == "__main__":
    main()