from __future__ import annotations

import pytest

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.request_state import RequestState


pytestmark = [pytest.mark.llama, pytest.mark.slow]


def test_custom_llama_decode_engine_uses_contiguous_kv_cache() -> None:
    engine = CustomLlamaDecodeEngine()

    request_state = RequestState(
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    engine.init_request_state(request_state)

    assert request_state.prompt_tokens == 6
    assert request_state.generated_tokens == 0
    assert request_state.input_ids is not None
    assert request_state.past_key_values is not None
    assert request_state.next_token is not None

    assert tuple(request_state.input_ids.shape) == (1, 6)
    assert len(request_state.past_key_values) == engine.config.num_hidden_layers

    layer0_key, layer0_value = request_state.past_key_values[0]
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
            kv_block_manager=None,  # type: ignore[arg-type]
        )

        assert output.decode_batch_snapshot is not None
        assert output.decode_batch_snapshot["backend"] == "custom-llama-contiguous-kv-cache"
        assert output.decode_batch_snapshot["uses_kv_cache"] is True
        assert output.decode_batch_snapshot["uses_paged_attention"] is False

        request_output = output.request_outputs[0]

        text_pieces.append(request_output.text)

        assert request_output.generated_tokens == 1
        assert request_state.generated_tokens == expected_generated_tokens
        assert tuple(request_state.input_ids.shape) == (
            1,
            request_state.prompt_tokens + expected_generated_tokens,
        )

        if not request_output.finished:
            assert request_state.past_key_values is not None
            layer0_key, layer0_value = request_state.past_key_values[0]
            assert tuple(layer0_key.shape) == (
                1,
                engine.num_key_value_heads,
                request_state.prompt_tokens + expected_generated_tokens,
                engine.head_dim,
            )
            assert tuple(layer0_value.shape) == (
                1,
                engine.num_key_value_heads,
                request_state.prompt_tokens + expected_generated_tokens,
                engine.head_dim,
            )

        if request_output.finished:
            break

    generated_text = "".join(text_pieces)

    print(f"text_pieces={text_pieces}")
    print(f"generated_text={generated_text!r}")
    print(f"generated_tokens={request_state.generated_tokens}")
    print(f"input_ids shape={tuple(request_state.input_ids.shape)}")

    assert generated_text == "Paris.\n\n"
    assert request_state.generated_tokens == 4