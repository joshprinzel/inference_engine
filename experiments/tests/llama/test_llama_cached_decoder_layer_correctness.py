from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.attention import build_causal_mask, build_decode_attention_mask
from engines.llama.cached_layer import llama_decoder_layer_forward_with_kv_cache
from engines.llama.layer import llama_decoder_layer_forward


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"
NEXT_TEXT = " Paris"

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


def run_custom_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    config: object,
) -> torch.Tensor:
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    attn = layer.self_attn
    mlp = layer.mlp

    return llama_decoder_layer_forward(
        hidden_states=hidden_states,
        input_layernorm_weight=layer.input_layernorm.weight,
        post_attention_layernorm_weight=layer.post_attention_layernorm.weight,
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
        attention_scaling=getattr(attn, "scaling", head_dim**-0.5),
    )


def run_custom_cached_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    config: object,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    attn = layer.self_attn
    mlp = layer.mlp

    return llama_decoder_layer_forward_with_kv_cache(
        hidden_states=hidden_states,
        input_layernorm_weight=layer.input_layernorm.weight,
        post_attention_layernorm_weight=layer.post_attention_layernorm.weight,
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
        past_key=past_key,
        past_value=past_value,
        q_proj_bias=attn.q_proj.bias,
        k_proj_bias=attn.k_proj.bias,
        v_proj_bias=attn.v_proj.bias,
        o_proj_bias=attn.o_proj.bias,
        gate_proj_bias=mlp.gate_proj.bias,
        up_proj_bias=mlp.up_proj.bias,
        down_proj_bias=mlp.down_proj.bias,
        attention_scaling=getattr(attn, "scaling", head_dim**-0.5),
    )


def test_cached_decoder_layer_matches_full_recompute_last_token() -> None:
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

    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    # Encode NEXT_TEXT without adding a BOS token.
    next_ids = tokenizer(
        NEXT_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    next_ids = next_ids[:, :1]

    full_ids = torch.cat([prompt_ids, next_ids], dim=-1)

    batch_size, prompt_len = prompt_ids.shape
    _, full_len = full_ids.shape

    hidden_size = config.hidden_size
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    head_dim = hidden_size // num_attention_heads

    layer0 = model.model.layers[0]

    with torch.inference_mode():
        prompt_hidden = model.model.embed_tokens(prompt_ids)
        full_hidden = model.model.embed_tokens(full_ids)
        decode_hidden = model.model.embed_tokens(next_ids)

        # Full recompute path.
        full_position_ids = torch.arange(
            0,
            full_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        full_rope_x = torch.empty(
            batch_size,
            num_key_value_heads,
            full_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        full_cos, full_sin = call_hf_rotary_emb(
            model.model.rotary_emb,
            full_rope_x,
            full_position_ids,
        )

        full_attention_mask = build_causal_mask(
            batch_size=batch_size,
            q_len=full_len,
            kv_len=full_len,
            device=device,
            dtype=dtype,
        )

        full_layer_out = run_custom_layer(
            layer=layer0,
            hidden_states=full_hidden,
            cos=full_cos,
            sin=full_sin,
            attention_mask=full_attention_mask,
            config=config,
        )

        expected_last = full_layer_out[:, -1:, :]

        # Prefill cached path.
        prompt_position_ids = torch.arange(
            0,
            prompt_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

        prompt_rope_x = torch.empty(
            batch_size,
            num_key_value_heads,
            prompt_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        prompt_cos, prompt_sin = call_hf_rotary_emb(
            model.model.rotary_emb,
            prompt_rope_x,
            prompt_position_ids,
        )

        prompt_attention_mask = build_decode_attention_mask(
            batch_size=batch_size,
            q_len=prompt_len,
            kv_len=prompt_len,
            past_len=0,
            device=device,
            dtype=dtype,
        )

        prompt_out, past_key, past_value = run_custom_cached_layer(
            layer=layer0,
            hidden_states=prompt_hidden,
            cos=prompt_cos,
            sin=prompt_sin,
            attention_mask=prompt_attention_mask,
            config=config,
            past_key=None,
            past_value=None,
        )

        assert prompt_out.shape == prompt_hidden.shape
        assert past_key.shape == (batch_size, num_key_value_heads, prompt_len, head_dim)
        assert past_value.shape == (batch_size, num_key_value_heads, prompt_len, head_dim)

        # One-token decode path.
        decode_position_ids = torch.tensor(
            [[prompt_len]],
            device=device,
            dtype=torch.long,
        )

        decode_rope_x = torch.empty(
            batch_size,
            num_key_value_heads,
            1,
            head_dim,
            device=device,
            dtype=dtype,
        )

        decode_cos, decode_sin = call_hf_rotary_emb(
            model.model.rotary_emb,
            decode_rope_x,
            decode_position_ids,
        )

        decode_attention_mask = build_decode_attention_mask(
            batch_size=batch_size,
            q_len=1,
            kv_len=prompt_len + 1,
            past_len=prompt_len,
            device=device,
            dtype=dtype,
        )

        cached_last, present_key, present_value = run_custom_cached_layer(
            layer=layer0,
            hidden_states=decode_hidden,
            cos=decode_cos,
            sin=decode_sin,
            attention_mask=decode_attention_mask,
            config=config,
            past_key=past_key,
            past_value=past_value,
        )

    diff = (cached_last - expected_last).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_ids={prompt_ids.tolist()}")
    print(f"next_ids={next_ids.tolist()}")
    print(f"prompt_len={prompt_len}")
    print(f"full_len={full_len}")
    print(f"expected_last shape={tuple(expected_last.shape)}")
    print(f"cached_last shape={tuple(cached_last.shape)}")
    print(f"past_key shape={tuple(past_key.shape)}")
    print(f"present_key shape={tuple(present_key.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")

    assert cached_last.shape == expected_last.shape == (batch_size, 1, hidden_size)
    assert present_key.shape == (batch_size, num_key_value_heads, prompt_len + 1, head_dim)
    assert present_value.shape == (batch_size, num_key_value_heads, prompt_len + 1, head_dim)

    assert max_abs_error <= 2e-2
    assert mean_abs_error <= 2e-3