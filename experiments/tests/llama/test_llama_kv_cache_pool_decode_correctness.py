from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.kv_cache_transfer import (
    gather_past_key_values_from_pool,
    write_past_key_values_to_pool,
)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype}")





def test_decode_logits_match_when_using_kv_cache_pool_gathered_cache() -> None:
    device = resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)
    batch_size, prompt_len = prompt_ids.shape

    assert batch_size == 1

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads

    kv_block_manager = KVBlockManager(
        total_blocks=16,
        block_size_tokens=4,
    )

    block_table = kv_block_manager.allocate_for_tokens(
        request_id="req-0",
        num_tokens=prompt_len + 1,
    )

    layout = KVCacheLayout(
        num_layers=config.num_hidden_layers,
        total_blocks=kv_block_manager.total_blocks,
        block_size_tokens=kv_block_manager.block_size_tokens,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype_name(dtype),
        device=device,
    )

    kv_cache_pool = KVCachePool(layout)
    kv_cache_pool.zero_()

    with torch.inference_mode():
        prompt_logits, original_past_key_values = (
            llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=model,
                input_ids=prompt_ids,
                past_key_values=None,
            )
        )

        first_decode_token = torch.argmax(
            prompt_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        write_past_key_values_to_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            past_key_values=original_past_key_values,
        )

        gathered_past_key_values = gather_past_key_values_from_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            seq_len=prompt_len,
        )

        original_decode_logits, original_present_key_values = (
            llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=model,
                input_ids=first_decode_token,
                past_key_values=original_past_key_values,
            )
        )

        gathered_decode_logits, gathered_present_key_values = (
            llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=model,
                input_ids=first_decode_token,
                past_key_values=gathered_past_key_values,
            )
        )

    diff = (gathered_decode_logits - original_decode_logits).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    original_next_token = int(torch.argmax(original_decode_logits[:, -1, :], dim=-1).item())
    gathered_next_token = int(torch.argmax(gathered_decode_logits[:, -1, :], dim=-1).item())

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_ids={prompt_ids.tolist()}")
    print(f"prompt_len={prompt_len}")
    print(f"block_table={block_table}")
    print(f"first_decode_token={int(first_decode_token.item())}")
    print(f"original_decode_logits shape={tuple(original_decode_logits.shape)}")
    print(f"gathered_decode_logits shape={tuple(gathered_decode_logits.shape)}")
    print(f"original_layer0_present_key shape={tuple(original_present_key_values[0][0].shape)}")
    print(f"gathered_layer0_present_key shape={tuple(gathered_present_key_values[0][0].shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")
    print(f"original_next_token={original_next_token}")
    print(f"gathered_next_token={gathered_next_token}")

    assert original_decode_logits.shape == gathered_decode_logits.shape
    assert len(original_present_key_values) == len(gathered_present_key_values)

    for (original_key, original_value), (gathered_key, gathered_value) in zip(
        original_present_key_values,
        gathered_present_key_values,
    ):
        assert original_key.shape == gathered_key.shape
        assert original_value.shape == gathered_value.shape

    assert max_abs_error == 0.0
    assert mean_abs_error == 0.0
    assert original_next_token == gathered_next_token