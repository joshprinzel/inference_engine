from __future__ import annotations

import pytest

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.request_state import RequestState


pytestmark = [pytest.mark.llama, pytest.mark.slow]


def test_custom_llama_decode_engine_generates_real_tokens() -> None:
    engine = CustomLlamaDecodeEngine()

    request_state = RequestState(
        request_id="req-0",
        prompt="The capital of France is",
        max_new_tokens=4,
    )

    prompt_tokens = engine.count_prompt_tokens(request_state.prompt)
    assert prompt_tokens == 6

    engine.init_request_state(request_state)
    assert request_state.prompt_tokens == 6

    text_pieces: list[str] = []

    for _ in range(request_state.max_new_tokens):
        step_output = engine.decode_step(
            request_states=[request_state],
            kv_block_manager=None,  # type: ignore[arg-type]
        )

        assert len(step_output.request_outputs) == 1

        request_output = step_output.request_outputs[0]
        text_pieces.append(request_output.text)

        if request_output.finished:
            break

    generated_text = "".join(text_pieces)

    print(f"text_pieces={text_pieces}")
    print(f"generated_text={generated_text!r}")
    print(f"generated_tokens={request_state.generated_tokens}")

    assert generated_text.startswith("Paris")
    assert request_state.generated_tokens > 0