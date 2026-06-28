from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights
from engines.llama.projections import project_qkv_with_weights, reshape_qkv_for_attention
from engines.llama.rmsnorm import llama_rmsnorm
from engines.llama.rope import apply_llama_rope
from engines.llama.attention import llama_attention_with_kv_cache
from runtime.attention_backend import CudaPagedAttentionBackend
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.kv_cache_transfer import (
    gather_past_key_values_from_pool,
    write_past_key_values_to_pool,
)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA paged attention test")
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
    padded = []

    for row in block_tables:
        padded.append(row + [-1] * (max_blocks - len(row)))

    return torch.tensor(padded, device=device, dtype=torch.int32)


def compute_layer0_decode_q(
    model: torch.nn.Module,
    next_token: torch.Tensor,
    position_id: int,
) -> torch.Tensor:
    """
    Returns q_rot for layer 0 decode.

    Shape:
        [batch, num_query_heads, head_dim]
    """

    config = model.config
    layer0 = model.model.layers[0]
    attn = layer0.self_attn

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    hidden_states = model.model.embed_tokens(next_token)

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

    q_heads, k_heads, _ = reshape_qkv_for_attention(
        q=q,
        k=k,
        v=v,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )

    position_ids = torch.tensor(
        [[position_id]],
        device=next_token.device,
        dtype=torch.long,
    )

    rope_x = torch.empty(
        next_token.shape[0],
        num_key_value_heads,
        1,
        head_dim,
        device=next_token.device,
        dtype=hidden_states.dtype,
    )

    cos, sin = model.model.rotary_emb(rope_x, position_ids)

    q_rot, _ = apply_llama_rope(
        q=q_heads,
        k=k_heads,
        cos=cos,
        sin=sin,
        unsqueeze_dim=1,
    )

    return q_rot[:, :, 0, :].contiguous()


def test_tinyllama_layer0_cuda_paged_attention_matches_pytorch_attention() -> None:
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
    assert prompt_len < 512

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads

    kv_block_manager = KVBlockManager(
        total_blocks=64,
        block_size_tokens=16,
    )

    block_table = kv_block_manager.allocate_for_tokens(
        request_id="req-0",
        num_tokens=prompt_len,
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
        prompt_logits, prompt_past_key_values = (
            llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=model,
                input_ids=prompt_ids,
                past_key_values=None,
            )
        )

        next_token = torch.argmax(prompt_logits[:, -1, :], dim=-1, keepdim=True)

        write_past_key_values_to_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            past_key_values=prompt_past_key_values,
            start_token_position=0,
        )

        gathered_past_key_values = gather_past_key_values_from_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            seq_len=prompt_len,
        )

        q_decode = compute_layer0_decode_q(
            model=model,
            next_token=next_token,
            position_id=prompt_len,
        )

        layer0_past_key, layer0_past_value = gathered_past_key_values[0]

        # PyTorch/reference attention output before o_proj.
        reference_attn_out, _, _ = llama_attention_with_kv_cache(
            q=q_decode.unsqueeze(2),
            k=torch.empty(
                1,
                config.num_key_value_heads,
                0,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            v=torch.empty(
                1,
                config.num_key_value_heads,
                0,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            past_key=layer0_past_key,
            past_value=layer0_past_value,
            num_key_value_groups=config.num_attention_heads // config.num_key_value_heads,
            attention_mask=None,
            scaling=getattr(model.model.layers[0].self_attn, "scaling", head_dim**-0.5),
        )

        reference_attn_out = reference_attn_out[:, :, 0, :].contiguous()

        block_tables = build_block_tables_tensor(
            [block_table],
            device=device,
        )

        seq_lens = torch.tensor(
            [prompt_len],
            device=device,
            dtype=torch.int32,
        )

        cuda_backend = CudaPagedAttentionBackend()

        cuda_attn_out = cuda_backend.decode(
            q=q_decode,
            cache_pool=kv_cache_pool,
            layer_id=0,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )

    diff = (cuda_attn_out - reference_attn_out).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_len={prompt_len}")
    print(f"block_table={block_table}")
    print(f"q_decode shape={tuple(q_decode.shape)}")
    print(f"reference_attn_out shape={tuple(reference_attn_out.shape)}")
    print(f"cuda_attn_out shape={tuple(cuda_attn_out.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")

    assert q_decode.shape == (1, config.num_attention_heads, head_dim)
    assert cuda_attn_out.shape == reference_attn_out.shape
    assert max_abs_error <= 2e-2
    assert mean_abs_error <= 2e-3