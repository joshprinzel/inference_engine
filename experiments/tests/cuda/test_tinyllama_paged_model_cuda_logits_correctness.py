from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights
from engines.llama.paged_model import llama_model_decode_with_paged_attention_from_hf_weights
from runtime.attention_backend import CudaPagedAttentionBackend
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.kv_cache_transfer import write_past_key_values_to_pool


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA paged model decode test")
    return "cuda"


def dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_block_tables_tensor(
    block_tables: list[list[int]],
    device: str,
) -> torch.Tensor:
    max_blocks = max(len(row) for row in block_tables)

    padded: list[list[int]] = []
    for row in block_tables:
        padded.append(row + [-1] * (max_blocks - len(row)))

    return torch.tensor(padded, device=device, dtype=torch.int32)


def test_tinyllama_paged_model_cuda_decode_logits_match_cached_pytorch_decode() -> None:
    device = resolve_device()
    dtype = torch.float16

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
    assert prompt_len + 1 < 512

    config = model.config
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    block_table = kv_block_manager.allocate_for_tokens(
        request_id="req-0",
        num_tokens=prompt_len + 1,
    )

    layout = KVCacheLayout(
        num_layers=config.num_hidden_layers,
        total_blocks=kv_block_manager.total_blocks,
        block_size_tokens=kv_block_manager.block_size_tokens,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype_name(dtype),
        device=device,
    )

    kv_cache_pool = KVCachePool(layout)
    kv_cache_pool.zero_()

    block_tables_tensor = build_block_tables_tensor(
        [block_table],
        device=device,
    )

    seq_lens = torch.tensor(
        [prompt_len + 1],
        device=device,
        dtype=torch.int32,
    )

    cuda_backend = CudaPagedAttentionBackend()

    with torch.inference_mode():
        prompt_logits, prompt_past_key_values = (
            llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=model,
                input_ids=prompt_ids,
                past_key_values=None,
            )
        )

        next_token = torch.argmax(
            prompt_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        cached_decode_logits, cached_present_key_values = (
            llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=model,
                input_ids=next_token,
                past_key_values=prompt_past_key_values,
            )
        )

        write_past_key_values_to_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            past_key_values=prompt_past_key_values,
            start_token_position=0,
        )

        paged_decode_logits = llama_model_decode_with_paged_attention_from_hf_weights(
            hf_model=model,
            input_ids=next_token,
            token_position=prompt_len,
            block_table=block_table,
            block_tables_tensor=block_tables_tensor,
            seq_lens=seq_lens,
            kv_cache_pool=kv_cache_pool,
            attention_backend=cuda_backend,
        )

    diff = (paged_decode_logits - cached_decode_logits).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    cached_next_token = int(torch.argmax(cached_decode_logits[:, -1, :], dim=-1).item())
    paged_next_token = int(torch.argmax(paged_decode_logits[:, -1, :], dim=-1).item())

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_ids={prompt_ids.tolist()}")
    print(f"prompt_len={prompt_len}")
    print(f"next_token={int(next_token.item())}")
    print(f"block_table={block_table}")
    print(f"cached_decode_logits shape={tuple(cached_decode_logits.shape)}")
    print(f"paged_decode_logits shape={tuple(paged_decode_logits.shape)}")
    print(f"cached_present_layer0_key shape={tuple(cached_present_key_values[0][0].shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")
    print(f"cached_next_token={cached_next_token}")
    print(f"paged_next_token={paged_next_token}")

    assert cached_decode_logits.shape == paged_decode_logits.shape == (
        1,
        1,
        config.vocab_size,
    )

    assert max_abs_error <= 3e-2
    assert mean_abs_error <= 3e-3
    assert cached_next_token == paged_next_token