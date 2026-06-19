from __future__ import annotations

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from engines.llama.attention import build_causal_mask
from engines.llama.model import llama_model_forward


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8

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


def custom_full_forward_logits(
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

    dummy_rope_x = torch.empty(
        batch_size,
        num_key_value_heads,
        seq_len,
        head_dim,
        device=input_ids.device,
        dtype=dtype,
    )

    cos, sin = call_hf_rotary_emb(
        rotary_emb=model.model.rotary_emb,
        x=dummy_rope_x,
        position_ids=position_ids,
    )

    logits = llama_model_forward(
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

    return logits


def greedy_generate_custom(
    model: torch.nn.Module,
    config: object,
    input_ids: torch.Tensor,
    dtype: torch.dtype,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, list[int]]:
    generated = input_ids.clone()
    new_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        logits = custom_full_forward_logits(
            model=model,
            config=config,
            input_ids=generated,
            dtype=dtype,
        )

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        next_token_id = int(next_token.item())

        generated = torch.cat([generated, next_token], dim=-1)
        new_token_ids.append(next_token_id)

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return generated, new_token_ids


def greedy_generate_hf_stepwise(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, list[int]]:
    generated = input_ids.clone()
    new_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=None,
            use_cache=False,
        )

        logits = outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        next_token_id = int(next_token.item())

        generated = torch.cat([generated, next_token], dim=-1)
        new_token_ids.append(next_token_id)

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return generated, new_token_ids


def test_custom_llama_greedy_generation_matches_hf_stepwise() -> None:
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
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        custom_generated, custom_new_token_ids = greedy_generate_custom(
            model=model,
            config=config,
            input_ids=input_ids,
            dtype=dtype,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_token_id,
        )

        hf_generated, hf_new_token_ids = greedy_generate_hf_stepwise(
            model=model,
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=eos_token_id,
        )

    custom_text = tokenizer.decode(custom_generated[0], skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_generated[0], skip_special_tokens=True)

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt={PROMPT!r}")
    print(f"input_ids={input_ids.tolist()}")
    print(f"custom_new_token_ids={custom_new_token_ids}")
    print(f"hf_new_token_ids={hf_new_token_ids}")
    print(f"custom_text={custom_text!r}")
    print(f"hf_text={hf_text!r}")

    assert custom_new_token_ids == hf_new_token_ids
    assert torch.equal(custom_generated, hf_generated)