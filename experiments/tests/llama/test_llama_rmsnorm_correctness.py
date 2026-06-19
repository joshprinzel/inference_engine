from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from engines.llama.rmsnorm import llama_rmsnorm


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.llama, pytest.mark.slow]

def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available else "cpu"

def test_llama_rmsnorm_matches_hf_layer0_input_layernorm() -> None:
    device = resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(input_ids)
        hf_out = model.model.layers[0].input_layernorm(hidden_states)

        weight = model.model.layers[0].input_layernorm.weight
        custom_out = llama_rmsnorm(
            hidden_states=hidden_states,
            weight=weight,
            eps=config.rms_norm_eps
        )

    assert custom_out.shape == hf_out.shape
    assert custom_out.dtype == hf_out.dtype
    assert custom_out.device == hf_out.device

    max_abs_error = (custom_out - hf_out).abs().max().item()
    mean_abs_error = (custom_out - hf_out).abs().mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"hidden_states shape={tuple(hidden_states.shape)}")
    print(f"hf_out shape={tuple(hf_out.shape)}")
    print(f"custom_out shape={tuple(custom_out.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")

    assert max_abs_error <= 1e-3
    assert mean_abs_error <= 1e-4