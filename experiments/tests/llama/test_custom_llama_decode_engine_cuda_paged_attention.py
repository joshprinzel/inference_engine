from __future__ import annotations

import pytest
import torch

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.kv_block_manager import KVBlockManager
from runtime.request_state import RequestState


pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


def test_custom_llama_decode_engine_uses_cuda_paged_attention_single_step() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = CustomLlamaDecodeEngine(
        device="cuda",
        dtype=torch.float16,
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )

    request_state = RequestState(
        request_id="req-0",
        prompt="The capital of France is",
        max_new_tokens=2,
    )

    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    prompt_tokens = engine.count_prompt_tokens(request_state.prompt)

    request_state.block_table = kv_block_manager.allocate_for_tokens(
        request_id=request_state.request_id,
        num_tokens=prompt_tokens + request_state.max_new_tokens,
    )

    engine.init_request_state(request_state)

    assert request_state.prompt_tokens == prompt_tokens
    assert request_state.generated_tokens == 0
    assert request_state.next_token is not None
    assert int(request_state.next_token.item()) == 3681  # Paris

    output = engine.decode_step(
        request_states=[request_state],
        kv_block_manager=kv_block_manager,
    )

    assert len(output.request_outputs) == 1

    request_output = output.request_outputs[0]

    assert request_output.request_id == "req-0"
    assert request_output.generated_tokens == 1
    assert request_output.text == "Paris"
    assert request_output.finished is False

    assert request_state.generated_tokens == 1
    assert request_state.num_computed_tokens == prompt_tokens
    assert request_state.next_token is not None
    assert int(request_state.next_token.item()) == 29889  # "."

    assert output.decode_batch_snapshot["backend"] == "custom-llama-cuda-paged-attention-batched"
    assert output.decode_batch_snapshot["batched_decode"] == True
    assert output.decode_batch_snapshot["uses_kv_cache"] is True
    assert output.decode_batch_snapshot["uses_kv_cache_pool"] is True
    assert output.decode_batch_snapshot["uses_paged_attention"] is True
    assert output.decode_batch_snapshot["attention_backend"] == "cuda"


def test_custom_llama_decode_engine_cuda_paged_attention_multi_step_direct() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    engine = CustomLlamaDecodeEngine(
        device="cuda",
        dtype=torch.float16,
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )

    request_state = RequestState(
        request_id="req-0",
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    prompt_tokens = engine.count_prompt_tokens(request_state.prompt)

    request_state.block_table = kv_block_manager.allocate_for_tokens(
        request_id=request_state.request_id,
        num_tokens=prompt_tokens + request_state.max_new_tokens,
    )

    engine.init_request_state(request_state)

    pieces: list[str] = []

    for _ in range(4):
        output = engine.decode_step(
            request_states=[request_state],
            kv_block_manager=kv_block_manager,
        )

        request_output = output.request_outputs[0]
        pieces.append(request_output.text)

        if request_output.finished:
            break

    generated_text = "".join(pieces)

    print(f"generated_text={generated_text!r}")

    assert generated_text == "Paris.\n\n"
    assert request_state.generated_tokens == 4
    assert request_state.next_token is not None or request_state.is_finished()