from __future__ import annotations

import pytest

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_transfer import gather_past_key_values_from_pool
from runtime.request_state import RequestState


pytestmark = [pytest.mark.llama, pytest.mark.slow]


def test_custom_llama_decode_engine_uses_kv_cache_pool() -> None:
    engine = CustomLlamaDecodeEngine(
        total_kv_blocks=64,
        block_size_tokens=16,
    )

    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    request_state = RequestState(
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    prompt_tokens = engine.count_prompt_tokens(request_state.prompt)
    reserved_tokens = prompt_tokens + request_state.max_new_tokens

    request_state.block_table = kv_block_manager.allocate_for_tokens(
        request_id=str(request_state.request_id),
        num_tokens=reserved_tokens,
    )

    engine.init_request_state(request_state)

    assert request_state.prompt_tokens == 6
    assert request_state.generated_tokens == 0
    assert request_state.input_ids is not None
    assert request_state.past_key_values is None
    assert request_state.next_token is not None
    assert request_state.block_table is not None

    assert tuple(request_state.input_ids.shape) == (1, request_state.prompt_tokens)

    gathered_prompt_kv = gather_past_key_values_from_pool(
        kv_cache_pool=engine.kv_cache_pool,
        block_table=request_state.block_table,
        seq_len=request_state.prompt_tokens,
    )

    assert len(gathered_prompt_kv) == engine.config.num_hidden_layers

    layer0_key, layer0_value = gathered_prompt_kv[0]

    assert tuple(layer0_key.shape) == (
        1,
        engine.num_key_value_heads,
        request_state.prompt_tokens,
        engine.head_dim,
    )

    assert tuple(layer0_value.shape) == (
        1,
        engine.num_key_value_heads,
        request_state.prompt_tokens,
        engine.head_dim,
    )

    text_pieces: list[str] = []

    for expected_generated_tokens in range(1, request_state.max_new_tokens + 1):
        output = engine.decode_step(
            request_states=[request_state],
            kv_block_manager=kv_block_manager,
        )

        assert output.decode_batch_snapshot is not None
        assert output.decode_batch_snapshot["backend"] == "custom-llama-kv-cache-pool-gather"
        assert output.decode_batch_snapshot["uses_kv_cache"] is True
        assert output.decode_batch_snapshot["uses_kv_cache_pool"] is True
        assert output.decode_batch_snapshot["uses_paged_attention"] is False
        assert output.decode_batch_snapshot["kv_block_manager_present"] is True

        request_output = output.request_outputs[0]

        text_pieces.append(request_output.text)

        assert request_output.generated_tokens == 1
        assert request_state.generated_tokens == expected_generated_tokens
        assert request_state.past_key_values is None

        expected_seq_len = request_state.prompt_tokens + expected_generated_tokens

        assert tuple(request_state.input_ids.shape) == (
            1,
            expected_seq_len,
        )

        gathered_kv = gather_past_key_values_from_pool(
            kv_cache_pool=engine.kv_cache_pool,
            block_table=request_state.block_table,
            seq_len=expected_seq_len,
        )

        layer0_key, layer0_value = gathered_kv[0]

        assert tuple(layer0_key.shape) == (
            1,
            engine.num_key_value_heads,
            expected_seq_len,
            engine.head_dim,
        )

        assert tuple(layer0_value.shape) == (
            1,
            engine.num_key_value_heads,
            expected_seq_len,
            engine.head_dim,
        )

        if request_output.finished:
            break

    generated_text = "".join(text_pieces)

    print(f"text_pieces={text_pieces}")
    print(f"generated_text={generated_text!r}")
    print(f"generated_tokens={request_state.generated_tokens}")
    print(f"input_ids shape={tuple(request_state.input_ids.shape)}")
    print(f"block_table={request_state.block_table}")
    print(f"kv_cache_pool={engine.kv_cache_pool.snapshot()}")

    assert generated_text == "Paris.\n\n"
    assert request_state.generated_tokens == 4
    assert request_state.next_token is None