from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.llama.attention import (
    build_decode_attention_mask,
    llama_attention_with_kv_cache,
    merge_attention_heads,
)
from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights
from engines.llama.mlp import llama_swiglu_mlp
from engines.llama.projections import project_qkv_with_weights, reshape_qkv_for_attention
from engines.llama.rmsnorm import llama_rmsnorm
from engines.llama.rope import apply_llama_rope
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


def test_tinyllama_cuda_paged_attention_matches_pytorch_attention_all_layers() -> None:
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
    num_key_value_groups = num_attention_heads // num_key_value_heads
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

    block_tables = build_block_tables_tensor(
        [block_table],
        device=device,
    )

    seq_lens = torch.tensor(
        [prompt_len + 1],
        device=device,
        dtype=torch.int32,
    )

    cuda_backend = CudaPagedAttentionBackend()

    layer_max_errors: list[float] = []
    layer_mean_errors: list[float] = []

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

        write_past_key_values_to_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            past_key_values=prompt_past_key_values,
            start_token_position=0,
        )

        hidden_states = model.model.embed_tokens(next_token)

        position_ids = torch.tensor(
            [[prompt_len]],
            device=device,
            dtype=torch.long,
        )

        rope_x = torch.empty(
            batch_size,
            num_key_value_heads,
            1,
            head_dim,
            device=device,
            dtype=dtype,
        )

        cos, sin = model.model.rotary_emb(rope_x, position_ids)

        decode_attention_mask = build_decode_attention_mask(
            batch_size=batch_size,
            q_len=1,
            kv_len=prompt_len + 1,
            past_len=prompt_len,
            device=device,
            dtype=dtype,
        )

        for layer_id, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            residual = hidden_states

            normed_hidden_states = llama_rmsnorm(
                hidden_states=hidden_states,
                weight=layer.input_layernorm.weight,
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

            q_heads, k_heads, v_heads = reshape_qkv_for_attention(
                q=q,
                k=k,
                v=v,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )

            q_rot, k_rot = apply_llama_rope(
                q=q_heads,
                k=k_heads,
                cos=cos,
                sin=sin,
                unsqueeze_dim=1,
            )

            # Write this layer's current-token K/V into the physical cache
            # before attention. Real decode attention attends over:
            # prompt tokens + current token.
            current_token_position = prompt_len

            kv_cache_pool.write_request_token(
                layer_id=layer_id,
                block_table=block_table,
                token_position=current_token_position,
                key=k_rot[0, :, 0, :].contiguous(),
                value=v_heads[0, :, 0, :].contiguous(),
            )

            gathered_key_values = gather_past_key_values_from_pool(
                kv_cache_pool=kv_cache_pool,
                block_table=block_table,
                seq_len=prompt_len,
            )

            past_key, past_value = gathered_key_values[layer_id]

            reference_attn_out, _, _ = llama_attention_with_kv_cache(
                q=q_rot,
                k=k_rot,
                v=v_heads,
                past_key=past_key,
                past_value=past_value,
                num_key_value_groups=num_key_value_groups,
                attention_mask=decode_attention_mask,
                scaling=getattr(attn, "scaling", head_dim**-0.5),
            )

            reference_attn_out = reference_attn_out[:, :, 0, :].contiguous()

            cuda_attn_out = cuda_backend.decode(
                q=q_rot[:, :, 0, :].contiguous(),
                cache_pool=kv_cache_pool,
                layer_id=layer_id,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )

            diff = (cuda_attn_out - reference_attn_out).abs()
            max_abs_error = diff.max().item()
            mean_abs_error = diff.mean().item()

            layer_max_errors.append(max_abs_error)
            layer_mean_errors.append(mean_abs_error)

            print(
                f"layer={layer_id} "
                f"max_abs_error={max_abs_error} "
                f"mean_abs_error={mean_abs_error}"
            )

            assert cuda_attn_out.shape == reference_attn_out.shape
            assert max_abs_error <= 2e-2
            assert mean_abs_error <= 2e-3

            merged_attn = merge_attention_heads(
                reference_attn_out.unsqueeze(2),
            )

            attn_out = F.linear(
                merged_attn,
                attn.o_proj.weight,
                attn.o_proj.bias,
            )

            hidden_states = residual + attn_out

            residual = hidden_states

            normed_hidden_states = llama_rmsnorm(
                hidden_states=hidden_states,
                weight=layer.post_attention_layernorm.weight,
                eps=config.rms_norm_eps,
            )

            mlp_out = llama_swiglu_mlp(
                hidden_states=normed_hidden_states,
                gate_proj_weight=mlp.gate_proj.weight,
                up_proj_weight=mlp.up_proj.weight,
                down_proj_weight=mlp.down_proj.weight,
                gate_proj_bias=mlp.gate_proj.bias,
                up_proj_bias=mlp.up_proj.bias,
                down_proj_bias=mlp.down_proj.bias,
            )

            hidden_states = residual + mlp_out

    global_max_error = max(layer_max_errors)
    global_mean_error = max(layer_mean_errors)

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_len={prompt_len}")
    print(f"block_table={block_table}")
    print(f"num_layers={config.num_hidden_layers}")
    print(f"global_max_error={global_max_error}")
    print(f"global_mean_error={global_mean_error}")

    assert len(layer_max_errors) == config.num_hidden_layers
    assert global_max_error <= 2e-2
    assert global_mean_error <= 2e-3