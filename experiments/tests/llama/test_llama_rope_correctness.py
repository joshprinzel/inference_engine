from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

from engines.llama.projections import (
    project_qkv_with_weights,
    reshape_qkv_for_attention,
)
from engines.llama.rmsnorm import llama_rmsnorm
from engines.llama.rope import apply_llama_rope, rotate_half


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def call_hf_rotary_emb(
    rotary_emb: torch.nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call HF LlamaRotaryEmbedding across minor Transformers API differences.

    Newer versions generally use:

        rotary_emb(x, position_ids)

    Older versions may use:

        rotary_emb(x, seq_len=...)

    This adapter keeps the test focused on RoPE correctness rather than
    Transformers version churn.
    """

    signature = inspect.signature(rotary_emb.forward)

    if "position_ids" in signature.parameters:
        return rotary_emb(x, position_ids)

    if "seq_len" in signature.parameters:
        seq_len = int(position_ids.shape[-1])
        return rotary_emb(x, seq_len=seq_len)

    return rotary_emb(x, position_ids)


def call_hf_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call HF apply_rotary_pos_emb across minor Transformers API differences.

    Common newer signature:

        apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)

    Some older versions do not take unsqueeze_dim.
    """

    signature = inspect.signature(hf_apply_rotary_pos_emb)

    kwargs = {}

    if "position_ids" in signature.parameters:
        kwargs["position_ids"] = position_ids

    if "unsqueeze_dim" in signature.parameters:
        kwargs["unsqueeze_dim"] = 1

    return hf_apply_rotary_pos_emb(q, k, cos, sin, **kwargs)


def test_rotate_half_basic_shape_and_values() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    out = rotate_half(x)

    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])

    assert out.shape == x.shape
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_llama_rope_matches_hf_layer0_rotary_embedding() -> None:
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

    batch_size, seq_len = input_ids.shape

    position_ids = torch.arange(
        0,
        seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    assert position_ids.shape == (batch_size, seq_len)

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(input_ids)

        normed_hidden_states = llama_rmsnorm(
            hidden_states=hidden_states,
            weight=layer0.input_layernorm.weight,
            eps=config.rms_norm_eps,
        )

        q, k, v = project_qkv_with_weights(
            hidden_states=normed_hidden_states,
            q_weight=attn.q_proj.weight,
            k_weight=attn.k_proj.weight,
            v_weight=attn.v_proj.weight,
            q_bias=attn.q_proj.bias,
            k_bias=attn.k_proj.bias,
            v_bias=attn.v_proj.bias,
        )

        q_heads, k_heads, v_heads = reshape_qkv_for_attention(
            q=q,
            k=k,
            v=v,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        cos, sin = call_hf_rotary_emb(
            rotary_emb=model.model.rotary_emb,
            x=v_heads,
            position_ids=position_ids,
        )

        custom_q_rot, custom_k_rot = apply_llama_rope(
            q=q_heads,
            k=k_heads,
            cos=cos,
            sin=sin,
            unsqueeze_dim=1,
        )

        hf_q_rot, hf_k_rot = call_hf_apply_rotary_pos_emb(
            q=q_heads,
            k=k_heads,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )

    assert q_heads.shape == (1, 32, 6, 64)
    assert k_heads.shape == (1, 4, 6, 64)
    assert v_heads.shape == (1, 4, 6, 64)

    assert custom_q_rot.shape == hf_q_rot.shape == q_heads.shape
    assert custom_k_rot.shape == hf_k_rot.shape == k_heads.shape

    q_diff = (custom_q_rot - hf_q_rot).abs()
    k_diff = (custom_k_rot - hf_k_rot).abs()

    q_max = q_diff.max().item()
    k_max = k_diff.max().item()

    q_mean = q_diff.mean().item()
    k_mean = k_diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"input_ids shape={tuple(input_ids.shape)}")
    print(f"position_ids shape={tuple(position_ids.shape)}")
    print(f"q_heads shape={tuple(q_heads.shape)}")
    print(f"k_heads shape={tuple(k_heads.shape)}")
    print(f"v_heads shape={tuple(v_heads.shape)}")
    print(f"cos shape={tuple(cos.shape)}")
    print(f"sin shape={tuple(sin.shape)}")
    print(f"custom_q_rot shape={tuple(custom_q_rot.shape)}")
    print(f"custom_k_rot shape={tuple(custom_k_rot.shape)}")
    print(f"q max_abs_error={q_max}, mean_abs_error={q_mean}")
    print(f"k max_abs_error={k_max}, mean_abs_error={k_mean}")

    assert q_max == 0.0
    assert k_max == 0.0
    assert q_mean == 0.0
    assert k_mean == 0.0