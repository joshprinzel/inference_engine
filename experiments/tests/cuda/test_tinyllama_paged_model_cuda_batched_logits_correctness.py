from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights
from engines.llama.paged_model import (
    llama_model_decode_batch_with_paged_attention_from_hf_weights,
    llama_model_decode_with_paged_attention_from_hf_weights,
)
from runtime.attention_backend import CudaPagedAttentionBackend
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.kv_cache_transfer import write_past_key_values_to_pool


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


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

    padded = []
    for row in block_tables:
        padded.append(row + [-1] * (max_blocks - len(row)))

    return torch.tensor(
        padded,
        device=device,
        dtype=torch.int32,
    )


def test_tinyllama_batched_paged_model_cuda_logits_match_looped_single_request_logits() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    prompts = [
        "The capital of France is",
        "The capital of Germany is",
    ]

    encoded_inputs = [
        tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        for prompt in prompts
    ]

    prompt_lens = [int(input_ids.shape[-1]) for input_ids in encoded_inputs]

    assert len(set(prompt_lens)) == 1

    config = model.config
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    total_kv_blocks = 64
    block_size_tokens = 16

    kv_block_manager_single = KVBlockManager(
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
    )

    kv_block_manager_batch = KVBlockManager(
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
    )

    layout_single = KVCacheLayout(
        num_layers=config.num_hidden_layers,
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype_name(dtype),
        device=device,
    )

    layout_batch = KVCacheLayout(
        num_layers=config.num_hidden_layers,
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype_name(dtype),
        device=device,
    )

    kv_cache_pool_single = KVCachePool(layout_single)
    kv_cache_pool_batch = KVCachePool(layout_batch)

    kv_cache_pool_single.zero_()
    kv_cache_pool_batch.zero_()

    attention_backend = CudaPagedAttentionBackend()

    single_logits_by_request: list[torch.Tensor] = []
    batched_next_tokens: list[torch.Tensor] = []
    batched_token_positions: list[int] = []
    batched_block_tables: list[list[int]] = []

    with torch.inference_mode():
        for request_index, input_ids in enumerate(encoded_inputs):
            prompt_len = int(input_ids.shape[-1])

            prompt_logits, prompt_past_key_values = (
                llama_model_forward_with_kv_cache_from_hf_weights(
                    hf_model=model,
                    input_ids=input_ids,
                    past_key_values=None,
                )
            )

            next_token = torch.argmax(
                prompt_logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            block_table_single = kv_block_manager_single.allocate_for_tokens(
                request_id=f"single-{request_index}",
                num_tokens=prompt_len + 1,
            )

            write_past_key_values_to_pool(
                kv_cache_pool=kv_cache_pool_single,
                block_table=block_table_single,
                past_key_values=prompt_past_key_values,
                start_token_position=0,
            )

            single_block_tables_tensor = build_block_tables_tensor(
                [block_table_single],
                device=device,
            )

            single_seq_lens = torch.tensor(
                [prompt_len + 1],
                device=device,
                dtype=torch.int32,
            )

            single_logits = llama_model_decode_with_paged_attention_from_hf_weights(
                hf_model=model,
                input_ids=next_token,
                token_position=prompt_len,
                block_table=block_table_single,
                block_tables_tensor=single_block_tables_tensor,
                seq_lens=single_seq_lens,
                kv_cache_pool=kv_cache_pool_single,
                attention_backend=attention_backend,
            )

            single_logits_by_request.append(single_logits.detach().clone())

            block_table_batch = kv_block_manager_batch.allocate_for_tokens(
                request_id=f"batch-{request_index}",
                num_tokens=prompt_len + 1,
            )

            write_past_key_values_to_pool(
                kv_cache_pool=kv_cache_pool_batch,
                block_table=block_table_batch,
                past_key_values=prompt_past_key_values,
                start_token_position=0,
            )

            batched_next_tokens.append(next_token)
            batched_token_positions.append(prompt_len)
            batched_block_tables.append(block_table_batch)

        batched_input_ids = torch.cat(batched_next_tokens, dim=0)

        batched_block_tables_tensor = build_block_tables_tensor(
            batched_block_tables,
            device=device,
        )

        batched_seq_lens = torch.tensor(
            [position + 1 for position in batched_token_positions],
            device=device,
            dtype=torch.int32,
        )

        batched_logits = llama_model_decode_batch_with_paged_attention_from_hf_weights(
            hf_model=model,
            input_ids=batched_input_ids,
            token_positions=batched_token_positions,
            block_tables=batched_block_tables,
            block_tables_tensor=batched_block_tables_tensor,
            seq_lens=batched_seq_lens,
            kv_cache_pool=kv_cache_pool_batch,
            attention_backend=attention_backend,
        )

    assert batched_logits.shape == (
        len(prompts),
        1,
        config.vocab_size,
    )

    for request_index, single_logits in enumerate(single_logits_by_request):
        batched_row_logits = batched_logits[request_index : request_index + 1]

        diff = (batched_row_logits - single_logits).abs()
        max_abs_error = diff.max().item()
        mean_abs_error = diff.mean().item()

        single_next = int(torch.argmax(single_logits[:, -1, :], dim=-1).item())
        batched_next = int(torch.argmax(batched_row_logits[:, -1, :], dim=-1).item())

        print(
            f"request_index={request_index} "
            f"prompt={prompts[request_index]!r} "
            f"max_abs_error={max_abs_error} "
            f"mean_abs_error={mean_abs_error} "
            f"single_next={single_next} "
            f"batched_next={batched_next}"
        )

        assert max_abs_error <= 3e-2
        assert mean_abs_error <= 3e-3
        assert single_next == batched_next