from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

from engines.llama.attention import (
    build_causal_mask,
    llama_attention,
    llama_attention_with_output_projection,
    merge_attention_heads,
    repeat_kv,
)
from engines.llama.projections import (
    project_qkv_with_weights,
    reshape_qkv_for_attention,
)
from engines.llama.rmsnorm import llama_rmsnorm
from engines.llama.rope import apply_llama_rope


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
    signature = inspect.signature(rotary_emb.forward)

    if "position_ids" in signature.parameters:
        return rotary_emb(x, position_ids)

    if "seq_len" in signature.parameters:
        seq_len = int(position_ids.shape[-1])
        return rotary_emb(x, seq_len=seq_len)

    return rotary_emb(x, position_ids)


def call_hf_self_attn(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Call HF LlamaAttention across Transformers API variants.

    Newer Transformers versions pass RoPE cos/sin as position_embeddings.
    Older versions may use position_ids.
    """

    signature = inspect.signature(attn.forward)
    kwargs = {}

    if "attention_mask" in signature.parameters:
        kwargs["attention_mask"] = attention_mask

    if "position_embeddings" in signature.parameters:
        kwargs["position_embeddings"] = position_embeddings
    elif "position_ids" in signature.parameters:
        kwargs["position_ids"] = position_ids

    if "past_key_value" in signature.parameters:
        kwargs["past_key_value"] = None

    if "past_key_values" in signature.parameters:
        kwargs["past_key_values"] = None

    if "output_attentions" in signature.parameters:
        kwargs["output_attentions"] = False

    if "use_cache" in signature.parameters:
        kwargs["use_cache"] = False

    if "cache_position" in signature.parameters:
        cache_position = torch.arange(
            0,
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        )
        kwargs["cache_position"] = cache_position

    out = attn(hidden_states, **kwargs)

    if isinstance(out, tuple):
        return out[0]

    return out

def test_repeat_kv_shapes_and_values() -> None:
    x = torch.tensor(
        [
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ]
        ]
    )
    # x shape: [1, 2 kv heads, 1 seq, 2 dim]

    out = repeat_kv(x, num_key_value_groups=3)

    assert out.shape == (1, 6, 1, 2)

    expected = torch.tensor(
        [
            [
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
                [[3.0, 4.0]],
            ]
        ]
    )

    assert torch.equal(out, expected)


def test_llama_attention_matches_hf_layer0_self_attention() -> None:
    device = resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)

    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = getattr(attn, "scaling", head_dim**-0.5)

    batch_size, seq_len = input_ids.shape

    position_ids = torch.arange(
        0,
        seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    attention_mask = build_causal_mask(
        batch_size=batch_size,
        q_len=seq_len,
        kv_len=seq_len,
        device=input_ids.device,
        dtype=dtype,
    )

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

        q_rot, k_rot = apply_llama_rope(
            q=q_heads,
            k=k_heads,
            cos=cos,
            sin=sin,
            unsqueeze_dim=1,
        )

        custom_pre_o_proj = llama_attention(
            q=q_rot,
            k=k_rot,
            v=v_heads,
            num_key_value_groups=num_key_value_groups,
            attention_mask=attention_mask,
            scaling=scaling,
        )

        custom_merged = merge_attention_heads(custom_pre_o_proj)

        custom_attn_out = llama_attention_with_output_projection(
            q=q_rot,
            k=k_rot,
            v=v_heads,
            o_proj_weight=attn.o_proj.weight,
            o_proj_bias=attn.o_proj.bias,
            num_key_value_groups=num_key_value_groups,
            attention_mask=attention_mask,
            scaling=scaling,
        )

        hf_attn_out = call_hf_self_attn(
            attn=attn,
            hidden_states=normed_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )

    assert q_rot.shape == (1, 32, 6, 64)
    assert k_rot.shape == (1, 4, 6, 64)
    assert v_heads.shape == (1, 4, 6, 64)

    assert custom_pre_o_proj.shape == (1, 32, 6, 64)
    assert custom_merged.shape == (1, 6, 2048)

    assert custom_attn_out.shape == hf_attn_out.shape == (1, 6, 2048)
    assert custom_attn_out.dtype == hf_attn_out.dtype

    diff = (custom_attn_out - hf_attn_out).abs()

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"input_ids shape={tuple(input_ids.shape)}")
    print(f"position_ids shape={tuple(position_ids.shape)}")
    print(f"attention_mask shape={tuple(attention_mask.shape)}")
    print(f"hidden_states shape={tuple(hidden_states.shape)}")
    print(f"normed_hidden_states shape={tuple(normed_hidden_states.shape)}")
    print(f"q_rot shape={tuple(q_rot.shape)}")
    print(f"k_rot shape={tuple(k_rot.shape)}")
    print(f"v_heads shape={tuple(v_heads.shape)}")
    print(f"custom_pre_o_proj shape={tuple(custom_pre_o_proj.shape)}")
    print(f"custom_merged shape={tuple(custom_merged.shape)}")
    print(f"custom_attn_out shape={tuple(custom_attn_out.shape)}")
    print(f"attn implementation={getattr(model.config, '_attn_implementation', None)}")
    print(f"attn scaling={scaling}")
    print(f"hf_attn_out shape={tuple(hf_attn_out.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")


    assert max_abs_error <= 2e-3
    assert mean_abs_error <= 2e-4