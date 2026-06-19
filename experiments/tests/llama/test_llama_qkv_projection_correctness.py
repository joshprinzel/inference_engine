from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.projections import (
    project_qkv_with_weights,
    reshape_qkv_for_attention,
)
from engines.llama.rmsnorm import llama_rmsnorm


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_llama_qkv_projection_with_raw_weights_matches_hf_layer0_modules() -> None:
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
    attn = layer0.self_attn

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(input_ids)

        normed_hidden_states = llama_rmsnorm(
            hidden_states=hidden_states,
            weight=layer0.input_layernorm.weight,
            eps=config.rms_norm_eps,
        )

        custom_q, custom_k, custom_v = project_qkv_with_weights(
            hidden_states=normed_hidden_states,
            q_weight=attn.q_proj.weight,
            k_weight=attn.k_proj.weight,
            v_weight=attn.v_proj.weight,
            q_bias=attn.q_proj.bias,
            k_bias=attn.k_proj.bias,
            v_bias=attn.v_proj.bias,
        )

        hf_q = attn.q_proj(normed_hidden_states)
        hf_k = attn.k_proj(normed_hidden_states)
        hf_v = attn.v_proj(normed_hidden_states)

        custom_q_heads, custom_k_heads, custom_v_heads = reshape_qkv_for_attention(
            q=custom_q,
            k=custom_k,
            v=custom_v,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    assert custom_q.shape == hf_q.shape == (1, 6, 2048)
    assert custom_k.shape == hf_k.shape == (1, 6, 256)
    assert custom_v.shape == hf_v.shape == (1, 6, 256)

    assert custom_q_heads.shape == (1, 32, 6, 64)
    assert custom_k_heads.shape == (1, 4, 6, 64)
    assert custom_v_heads.shape == (1, 4, 6, 64)

    q_diff = (custom_q - hf_q).abs()
    k_diff = (custom_k - hf_k).abs()
    v_diff = (custom_v - hf_v).abs()

    q_max = q_diff.max().item()
    k_max = k_diff.max().item()
    v_max = v_diff.max().item()

    q_mean = q_diff.mean().item()
    k_mean = k_diff.mean().item()
    v_mean = v_diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"hidden_states shape={tuple(hidden_states.shape)}")
    print(f"normed_hidden_states shape={tuple(normed_hidden_states.shape)}")
    print(f"q weight shape={tuple(attn.q_proj.weight.shape)}")
    print(f"k weight shape={tuple(attn.k_proj.weight.shape)}")
    print(f"v weight shape={tuple(attn.v_proj.weight.shape)}")
    print(f"custom_q shape={tuple(custom_q.shape)}")
    print(f"custom_k shape={tuple(custom_k.shape)}")
    print(f"custom_v shape={tuple(custom_v.shape)}")
    print(f"custom_q_heads shape={tuple(custom_q_heads.shape)}")
    print(f"custom_k_heads shape={tuple(custom_k_heads.shape)}")
    print(f"custom_v_heads shape={tuple(custom_v_heads.shape)}")
    print(f"q max_abs_error={q_max}, mean_abs_error={q_mean}")
    print(f"k max_abs_error={k_max}, mean_abs_error={k_mean}")
    print(f"v max_abs_error={v_max}, mean_abs_error={v_mean}")

    assert q_max == 0.0
    assert k_max == 0.0
    assert v_max == 0.0