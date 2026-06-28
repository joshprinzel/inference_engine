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
    write_last_token_past_key_values_to_pool,
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





def test_kv_cache_pool_roundtrips_tinyllama_past_key_values() -> None:
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

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)
    batch_size, seq_len = input_ids.shape

    assert batch_size == 1

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads

    kv_block_manager = KVBlockManager(
        total_blocks=16,
        block_size_tokens=4,
    )

    block_table = kv_block_manager.allocate_for_tokens(
        request_id="req-0",
        num_tokens=seq_len,
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
        _, past_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
            hf_model=model,
            input_ids=input_ids,
            past_key_values=None,
        )

        write_past_key_values_to_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            past_key_values=past_key_values,
        )

        gathered_key_values = gather_past_key_values_from_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            seq_len=seq_len,
        )

    assert len(gathered_key_values) == len(past_key_values) == config.num_hidden_layers

    max_errors: list[float] = []
    mean_errors: list[float] = []

    for layer_id, ((expected_key, expected_value), (actual_key, actual_value)) in enumerate(
        zip(past_key_values, gathered_key_values)
    ):
        assert actual_key.shape == expected_key.shape
        assert actual_value.shape == expected_value.shape

        key_diff = (actual_key - expected_key).abs()
        value_diff = (actual_value - expected_value).abs()

        max_errors.append(max(key_diff.max().item(), value_diff.max().item()))
        mean_errors.append(max(key_diff.mean().item(), value_diff.mean().item()))

    global_max_error = max(max_errors)
    global_mean_error = max(mean_errors)

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"input_ids={input_ids.tolist()}")
    print(f"seq_len={seq_len}")
    print(f"block_table={block_table}")
    print(f"layout={layout.snapshot()}")
    print(f"kv_cache_pool={kv_cache_pool.snapshot()}")
    print(f"layer0_expected_key_shape={tuple(past_key_values[0][0].shape)}")
    print(f"layer0_gathered_key_shape={tuple(gathered_key_values[0][0].shape)}")
    print(f"global_max_error={global_max_error}")
    print(f"global_mean_error={global_mean_error}")

    assert global_max_error == 0.0
    assert global_mean_error == 0.0