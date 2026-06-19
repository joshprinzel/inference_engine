from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.mlp import llama_swiglu_mlp
from engines.llama.rmsnorm import llama_rmsnorm


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_llama_swiglu_mlp_matches_hf_layer0_mlp() -> None:
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

    layer0 = model.model.layers[0]
    mlp = layer0.mlp

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(input_ids)

        # This test isolates the MLP component using the same kind of input the
        # MLP receives inside a decoder layer: normalized hidden states.
        normed_hidden_states = llama_rmsnorm(
            hidden_states=hidden_states,
            weight=layer0.post_attention_layernorm.weight,
            eps=config.rms_norm_eps,
        )

        custom_out = llama_swiglu_mlp(
            hidden_states=normed_hidden_states,
            gate_proj_weight=mlp.gate_proj.weight,
            up_proj_weight=mlp.up_proj.weight,
            down_proj_weight=mlp.down_proj.weight,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
        )

        hf_out = mlp(normed_hidden_states)

    assert custom_out.shape == hf_out.shape == (1, 6, 2048)
    assert custom_out.dtype == hf_out.dtype
    assert custom_out.device == hf_out.device

    diff = (custom_out - hf_out).abs()

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"hidden_states shape={tuple(hidden_states.shape)}")
    print(f"normed_hidden_states shape={tuple(normed_hidden_states.shape)}")
    print(f"gate_proj weight shape={tuple(mlp.gate_proj.weight.shape)}")
    print(f"up_proj weight shape={tuple(mlp.up_proj.weight.shape)}")
    print(f"down_proj weight shape={tuple(mlp.down_proj.weight.shape)}")
    print(f"custom_out shape={tuple(custom_out.shape)}")
    print(f"hf_out shape={tuple(hf_out.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")

    assert max_abs_error == 0.0
    assert mean_abs_error == 0.0