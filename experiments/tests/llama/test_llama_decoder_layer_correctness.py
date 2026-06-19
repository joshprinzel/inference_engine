from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.attention import build_causal_mask
from engines.llama.layer import llama_decoder_layer_forward


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


def call_hf_decoder_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Call HF LlamaDecoderLayer across Transformers API variants.
    """

    signature = inspect.signature(layer.forward)
    kwargs = {}

    if "attention_mask" in signature.parameters:
        kwargs["attention_mask"] = attention_mask

    if "position_ids" in signature.parameters:
        kwargs["position_ids"] = position_ids

    if "position_embeddings" in signature.parameters:
        kwargs["position_embeddings"] = position_embeddings

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

    out = layer(hidden_states, **kwargs)

    if isinstance(out, tuple):
        return out[0]

    return out


def test_llama_decoder_layer0_matches_hf_decoder_layer0() -> None:
    device = resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

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

    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    attention_scaling = getattr(attn, "scaling", head_dim**-0.5)

    with torch.inference_mode():
        hidden_states = model.model.embed_tokens(input_ids)

        # HF rotary embedding only needs a tensor with compatible device/dtype/shape.
        # v_proj output shape would also work, but embedding output is enough for
        # calling model.model.rotary_emb in current Transformers versions.
        cos, sin = call_hf_rotary_emb(
            rotary_emb=model.model.rotary_emb,
            x=hidden_states,
            position_ids=position_ids,
        )

        custom_out = llama_decoder_layer_forward(
            hidden_states=hidden_states,
            input_layernorm_weight=layer0.input_layernorm.weight,
            post_attention_layernorm_weight=layer0.post_attention_layernorm.weight,
            q_proj_weight=attn.q_proj.weight,
            k_proj_weight=attn.k_proj.weight,
            v_proj_weight=attn.v_proj.weight,
            o_proj_weight=attn.o_proj.weight,
            gate_proj_weight=mlp.gate_proj.weight,
            up_proj_weight=mlp.up_proj.weight,
            down_proj_weight=mlp.down_proj.weight,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            rms_norm_eps=config.rms_norm_eps,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            q_proj_bias=attn.q_proj.bias,
            k_proj_bias=attn.k_proj.bias,
            v_proj_bias=attn.v_proj.bias,
            o_proj_bias=attn.o_proj.bias,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
            attention_scaling=attention_scaling,
        )

        hf_out = call_hf_decoder_layer(
            layer=layer0,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )

    assert custom_out.shape == hf_out.shape == (1, 6, 2048)
    assert custom_out.dtype == hf_out.dtype
    assert custom_out.device == hf_out.device

    diff = (custom_out - hf_out).abs()

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"input_ids shape={tuple(input_ids.shape)}")
    print(f"hidden_states shape={tuple(hidden_states.shape)}")
    print(f"position_ids shape={tuple(position_ids.shape)}")
    print(f"attention_mask shape={tuple(attention_mask.shape)}")
    print(f"cos shape={tuple(cos.shape)}")
    print(f"sin shape={tuple(sin.shape)}")
    print(f"attention_scaling={attention_scaling}")
    print(f"custom_out shape={tuple(custom_out.shape)}")
    print(f"hf_out shape={tuple(hf_out.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")

    assert max_abs_error <= 3e-3
    assert mean_abs_error <= 3e-4