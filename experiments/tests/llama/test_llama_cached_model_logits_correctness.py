from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.attention import build_causal_mask
from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights
from engines.llama.model import llama_model_forward


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


def custom_full_recompute_logits(
    model: torch.nn.Module,
    config: object,
    input_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    position_ids = torch.arange(
        0,
        seq_len,
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)

    attention_mask = build_causal_mask(
        batch_size=batch_size,
        q_len=seq_len,
        kv_len=seq_len,
        device=input_ids.device,
        dtype=dtype,
    )

    rope_x = torch.empty(
        batch_size,
        num_key_value_heads,
        seq_len,
        head_dim,
        device=input_ids.device,
        dtype=dtype,
    )

    cos, sin = call_hf_rotary_emb(
        rotary_emb=model.model.rotary_emb,
        x=rope_x,
        position_ids=position_ids,
    )

    return llama_model_forward(
        input_ids=input_ids,
        embed_tokens_weight=model.model.embed_tokens.weight,
        layers=list(model.model.layers),
        final_norm_weight=model.model.norm.weight,
        lm_head_weight=model.lm_head.weight,
        cos=cos,
        sin=sin,
        attention_mask=attention_mask,
        rms_norm_eps=config.rms_norm_eps,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )


def test_cached_model_decode_logits_match_full_recompute_last_token() -> None:
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

    with torch.inference_mode():
        full_logits = custom_full_recompute_logits(
            model=model,
            config=config,
            input_ids=full_ids,
            dtype=dtype,
        )

        expected_last_logits = full_logits[:, -1:, :]

        prompt_logits, past_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
            hf_model=model,
            input_ids=prompt_ids,
            past_key_values=None,
        )

        decode_logits, present_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
            hf_model=model,
            input_ids=next_ids,
            past_key_values=past_key_values,
        )

    diff = (decode_logits - expected_last_logits).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    custom_full_next_token = int(torch.argmax(expected_last_logits[:, -1, :], dim=-1).item())
    cached_next_token = int(torch.argmax(decode_logits[:, -1, :], dim=-1).item())

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_ids={prompt_ids.tolist()}")
    print(f"next_ids={next_ids.tolist()}")
    print(f"prompt_len={prompt_len}")
    print(f"full_len={full_len}")
    print(f"prompt_logits shape={tuple(prompt_logits.shape)}")
    print(f"decode_logits shape={tuple(decode_logits.shape)}")
    print(f"expected_last_logits shape={tuple(expected_last_logits.shape)}")
    print(f"num_cached_layers={len(past_key_values)}")
    print(f"layer0_past_key shape={tuple(past_key_values[0][0].shape)}")
    print(f"layer0_present_key shape={tuple(present_key_values[0][0].shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")
    print(f"custom_full_next_token={custom_full_next_token}")
    print(f"cached_next_token={cached_next_token}")

    assert prompt_logits.shape == (batch_size, prompt_len, config.vocab_size)
    assert decode_logits.shape == (batch_size, 1, config.vocab_size)
    assert expected_last_logits.shape == (batch_size, 1, config.vocab_size)

    assert len(past_key_values) == config.num_hidden_layers
    assert len(present_key_values) == config.num_hidden_layers

    for layer_index, (past_key, past_value) in enumerate(past_key_values):
        assert past_key.shape == (
            batch_size,
            num_key_value_heads,
            prompt_len,
            head_dim,
        )
        assert past_value.shape == (
            batch_size,
            num_key_value_heads,
            prompt_len,
            head_dim,
        )

    for layer_index, (present_key, present_value) in enumerate(present_key_values):
        assert present_key.shape == (
            batch_size,
            num_key_value_heads,
            prompt_len + 1,
            head_dim,
        )
        assert present_value.shape == (
            batch_size,
            num_key_value_heads,
            prompt_len + 1,
            head_dim,
        )

    assert max_abs_error <= 3e-2
    assert mean_abs_error <= 3e-3
    assert cached_next_token == custom_full_next_token