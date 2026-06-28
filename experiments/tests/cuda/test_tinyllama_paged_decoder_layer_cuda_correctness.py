from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines.llama.cached_layer import llama_decoder_layer_forward_with_kv_cache
from engines.llama.paged_layer import llama_decoder_layer_forward_with_paged_attention
from runtime.attention_backend import CudaPagedAttentionBackend
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.kv_cache_transfer import write_past_key_values_to_pool
from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


def resolve_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA paged decoder layer test")
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


def test_tinyllama_layer0_paged_decoder_layer_cuda_matches_cached_layer() -> None:
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

    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp

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

        write_past_key_values_to_pool(
            kv_cache_pool=kv_cache_pool,
            block_table=block_table,
            past_key_values=prompt_past_key_values,
            start_token_position=0,
        )

        decode_hidden_states = model.model.embed_tokens(next_token)

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

        past_key, past_value = prompt_past_key_values[0]

        cached_layer_out, cached_present_key, cached_present_value = (
            llama_decoder_layer_forward_with_kv_cache(
                hidden_states=decode_hidden_states,
                input_layernorm_weight=layer0.input_layernorm.weight,
                post_attention_layernorm_weight=layer0.post_attention_layernorm.weight,
                q_proj_weight=attn.q_proj.weight,
                k_proj_weight=attn.k_proj.weight,
                v_proj_weight=attn.v_proj.weight,
                o_proj_weight=attn.o_proj.weight,
                gate_proj_weight=mlp.gate_proj.weight,
                up_proj_weight=mlp.up_proj.weight,
                down_proj_weight=mlp.down_proj.weight,
                cos=cos,
                sin=sin,
                attention_mask=None,
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
        )

        paged_layer_out = llama_decoder_layer_forward_with_paged_attention(
            hidden_states=decode_hidden_states,
            input_layernorm_weight=layer0.input_layernorm.weight,
            post_attention_layernorm_weight=layer0.post_attention_layernorm.weight,
            q_proj_weight=attn.q_proj.weight,
            k_proj_weight=attn.k_proj.weight,
            v_proj_weight=attn.v_proj.weight,
            o_proj_weight=attn.o_proj.weight,
            gate_proj_weight=mlp.gate_proj.weight,
            up_proj_weight=mlp.up_proj.weight,
            down_proj_weight=mlp.down_proj.weight,
            cos=cos,
            sin=sin,
            rms_norm_eps=config.rms_norm_eps,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            layer_id=0,
            token_position=prompt_len,
            block_table=block_table,
            block_tables_tensor=block_tables_tensor,
            seq_lens=seq_lens,
            kv_cache_pool=kv_cache_pool,
            attention_backend=cuda_backend,
            q_proj_bias=attn.q_proj.bias,
            k_proj_bias=attn.k_proj.bias,
            v_proj_bias=attn.v_proj.bias,
            o_proj_bias=attn.o_proj.bias,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
        )

    diff = (paged_layer_out - cached_layer_out).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompt_len={prompt_len}")
    print(f"block_table={block_table}")
    print(f"decode_hidden_states shape={tuple(decode_hidden_states.shape)}")
    print(f"cached_layer_out shape={tuple(cached_layer_out.shape)}")
    print(f"paged_layer_out shape={tuple(paged_layer_out.shape)}")
    print(f"cached_present_key shape={tuple(cached_present_key.shape)}")
    print(f"cached_present_value shape={tuple(cached_present_value.shape)}")
    print(f"max_abs_error={max_abs_error}")
    print(f"mean_abs_error={mean_abs_error}")

    assert cached_layer_out.shape == paged_layer_out.shape == (1, 1, hidden_size)
    assert cached_present_key.shape == (
        1,
        num_key_value_heads,
        prompt_len + 1,
        head_dim,
    )
    assert cached_present_value.shape == (
        1,
        num_key_value_heads,
        prompt_len + 1,
        head_dim,
    )

    assert max_abs_error <= 2e-2
    assert mean_abs_error <= 2e-3