from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.attention import build_causal_mask
from engines.llama.model import llama_model_forward


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


def test_llama_full_model_logits_match_hf() -> None:
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

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

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

    dummy_rope_x = torch.empty(
        batch_size,
        num_key_value_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        cos, sin = call_hf_rotary_emb(
            rotary_emb=model.model.rotary_emb,
            x=dummy_rope_x,
            position_ids=position_ids,
        )

        custom_logits = llama_model_forward(
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

        hf_logits = model(input_ids=input_ids, attention_mask=None, use_cache=False).logits

    assert custom_logits.shape == hf_logits.shape == (1, seq_len, config.vocab_size)
    assert custom_logits.dtype == hf_logits.dtype
    assert custom_logits.device == hf_logits.device

    diff = (custom_logits - hf_logits).abs()

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    custom_next_token_id = int(torch.argmax(custom_logits[:, -1, :], dim=-1).item())
    hf_next_token_id = int(torch.argmax(hf_logits[:, -1, :], dim=-1).item())

    custom_next_text = tokenizer.decode([custom_next_token_id])
    hf_next_text = tokenizer.decode([hf_next_token_id])

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"input_ids shape={tuple(input_ids.shape)}")
    print(f"position_ids shape={tuple(position_ids.shape)}")
    print(f"attention_mask shape={tuple(attention_mask.shape)}")
    print(f"cos shape={tuple(cos.shape)}")
    print(f"sin shape={tuple(sin.shape)}")
    print(f"custom_logits shape={tuple(custom_logits.shape)}")
    print(f"hf_logits shape={tuple(hf_logits.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")
    print(f"custom_next_token_id={custom_next_token_id}")
    print(f"hf_next_token_id={hf_next_token_id}")
    print(f"custom_next_text={custom_next_text!r}")
    print(f"hf_next_text={hf_next_text!r}")

    assert max_abs_error <= 2e-2
    assert mean_abs_error <= 2e-3

    assert custom_next_token_id == hf_next_token_id