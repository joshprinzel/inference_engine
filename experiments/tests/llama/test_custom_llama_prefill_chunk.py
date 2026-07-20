import pytest
import torch

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.kv_block_manager import KVBlockManager
from runtime.request_state import RequestState


@pytest.mark.runtime
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Custom Llama chunk prefill smoke requires CUDA.",
)
def test_custom_llama_prefill_chunk_materializes_first_prompt_chunk() -> None:
    engine = CustomLlamaDecodeEngine(
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )
    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    request = RequestState(
        prompt=(
            "Explain why paged KV cache matters for LLM serving in one "
            "short paragraph."
        ),
        max_new_tokens=4,
        request_id="req-1",
    )

    prompt_tokens = engine.count_prompt_tokens(request.prompt)
    assert prompt_tokens > 4

    request.status = "prefill"
    request.prompt_tokens = prompt_tokens
    request.block_table = kv_block_manager.allocate_for_tokens(
        request_id=str(request.request_id),
        num_tokens=prompt_tokens + request.max_new_tokens,
    )

    engine.prefill_chunk(
        request_state=request,
        num_tokens=4,
        kv_block_manager=kv_block_manager,
    )

    assert request.status == "prefill"
    assert request.prompt_tokens == prompt_tokens
    assert request.num_computed_tokens == 4
    assert request.prefill_tokens_remaining == prompt_tokens - 4
    assert request.generated_tokens == 0
    assert request.next_token is None
    assert request.input_ids is not None


@pytest.mark.runtime
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Custom Llama chunk prefill smoke requires CUDA.",
)
def test_custom_llama_prefill_chunk_materializes_multiple_prompt_chunks() -> None:
    engine = CustomLlamaDecodeEngine(
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )
    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    request = RequestState(
        prompt=(
            "Explain why paged KV cache matters for LLM serving in one "
            "short paragraph."
        ),
        max_new_tokens=4,
        request_id="req-1",
    )

    prompt_tokens = engine.count_prompt_tokens(request.prompt)
    assert prompt_tokens > 8

    request.status = "prefill"
    request.prompt_tokens = prompt_tokens
    request.block_table = kv_block_manager.allocate_for_tokens(
        request_id=str(request.request_id),
        num_tokens=prompt_tokens + request.max_new_tokens,
    )

    engine.prefill_chunk(
        request_state=request,
        num_tokens=4,
        kv_block_manager=kv_block_manager,
    )

    assert request.num_computed_tokens == 4
    assert request.prefill_tokens_remaining == prompt_tokens - 4
    assert request.next_token is None
    assert request.past_key_values is not None

    engine.prefill_chunk(
        request_state=request,
        num_tokens=4,
        kv_block_manager=kv_block_manager,
    )

    assert request.num_computed_tokens == 8
    assert request.prefill_tokens_remaining == prompt_tokens - 8
    assert request.next_token is None
    assert request.past_key_values is not None


@pytest.mark.runtime
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Custom Llama chunk prefill smoke requires CUDA.",
)
def test_custom_llama_prefill_chunk_sets_next_token_after_final_chunk() -> None:
    engine = CustomLlamaDecodeEngine(
        attention_backend_name="cuda",
        total_kv_blocks=64,
        block_size_tokens=16,
    )
    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    request = RequestState(
        prompt="Explain paged KV cache briefly.",
        max_new_tokens=4,
        request_id="req-1",
    )

    prompt_tokens = engine.count_prompt_tokens(request.prompt)
    assert prompt_tokens > 1

    request.status = "prefill"
    request.prompt_tokens = prompt_tokens
    request.block_table = kv_block_manager.allocate_for_tokens(
        request_id=str(request.request_id),
        num_tokens=prompt_tokens + request.max_new_tokens,
    )

    chunk_size = max(1, prompt_tokens // 2)

    while request.prefill_tokens_remaining > 0:
        engine.prefill_chunk(
            request_state=request,
            num_tokens=chunk_size,
            kv_block_manager=kv_block_manager,
        )

    assert request.num_computed_tokens == prompt_tokens
    assert request.prefill_tokens_remaining == 0
    assert request.next_token is not None
    assert request.past_key_values is None