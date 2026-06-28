from __future__ import annotations

import torch
import torch.nn.functional as F

from engines.llama.paged_layer import(
    llama_decoder_layer_forward_batch_with_paged_attention,
    llama_decoder_layer_forward_with_paged_attention
)
from engines.llama.rmsnorm import llama_rmsnorm
from runtime.attention_backend import AttentionBackend
from runtime.kv_cache_pool import KVCachePool


def llama_model_decode_with_paged_attention_from_hf_weights(
    hf_model: torch.nn.Module,
    input_ids: torch.Tensor,
    token_position: int,
    block_table: list[int],
    block_tables_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_pool: KVCachePool,
    attention_backend: AttentionBackend,
) -> torch.Tensor:
    """
    One-token TinyLlama decode using KVCachePool + pluggable paged attention backend.

    This function assumes:
        - input_ids shape is [1, 1]
        - KVCachePool already contains all previous-token K/V
        - this function writes current-token K/V for each layer into KVCachePool
        - attention_backend.decode reads K/V from KVCachePool using block tables
        - seq_lens includes the current token, so for first decode after a prompt
          of length P, seq_lens = [P + 1]
    """
    config = hf_model.config
    model = hf_model.model

    batch_size, q_len = input_ids.shape

    if batch_size != 1:
        raise ValueError(f"paged model decode currently expects batch_size=1, got {batch_size}")
    if q_len != 1:
        raise ValueError(f"paged model decode currently expects q_len=1, got {q_len}")
    
    device = input_ids.device
    dtype = model.embed_tokens.weight.dtype

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads

    hidden_states = F.embedding(input_ids, model.embed_tokens.weight)

    position_ids = torch.tensor(
        [[token_position]],
        device=device,
        dtype=torch.long
    )

    rope_x = torch.empty(
        batch_size,
        num_key_value_heads,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )

    cos, sin = model.rotary_emb(rope_x, position_ids)

    for layer_id, layer in enumerate(model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        hidden_states = llama_decoder_layer_forward_with_paged_attention(
            hidden_states=hidden_states,
            input_layernorm_weight=layer.input_layernorm.weight,
            post_attention_layernorm_weight=layer.post_attention_layernorm.weight,
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
            layer_id=layer_id,
            token_position=token_position,
            block_table=block_table,
            block_tables_tensor=block_tables_tensor,
            seq_lens=seq_lens,
            kv_cache_pool=kv_cache_pool,
            attention_backend=attention_backend,
            q_proj_bias=attn.q_proj.bias,
            k_proj_bias=attn.k_proj.bias,
            v_proj_bias=attn.v_proj.bias,
            o_proj_bias=attn.o_proj.bias,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
        )
    
    hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=model.norm.weight,
        eps=config.rms_norm_eps
    )

    logits = F.linear(hidden_states, hf_model.lm_head.weight,bias=None)
    return logits



def llama_model_decode_batch_with_paged_attention_from_hf_weights(
        hf_model: torch.nn.Module,
        input_ids: torch.Tensor,
        token_positions: list[int],
        block_tables: list[list[int]],
        block_tables_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache_pool: KVCachePool,
        attention_backend: AttentionBackend
) -> torch.Tensor:
    """
    Batched one-token Tiny Llama decode using KVCachePool + paged attention

    This is the batached equivalent to:
    llama_model_decode_with_paged_attention_from_hf_weights.

    Assumptions:
        - input_ids shape [batch, 1]
        - each request has exactly one decode token
        - KVCachePool already contains previous-token K/V for each request
        - this function writes each request's current-token K/V into KVCachePool
        - seq_lens includes the current token for each request
    """

    config = hf_model.config
    model = hf_model.model

    batch_size, q_len = input_ids.shape

    if q_len != 1:
        raise ValueError("batch paged model decode expects q_len=1, got {q_len}")
    
    if batch_size != len(token_positions):
        raise ValueError(
            f"batch_size={batch_size} does not match len(token_positions)={len(token_positions)}"
        )
    
    if batch_size != len(block_tables):
        raise ValueError(
            f"batch_size = {batch_size} does not match len(block_tables)={len(block_tables)}"
        )
    if block_tables_tensor.shape[0] != batch_size:
        raise ValueError(
            f"block_tables_tensor batch={block_tables_tensor.shape[0]} does not match batch_size = {batch_size}"
        )
    
    if seq_lens.shape[0] != batch_size:
        raise ValueError(
            f"seq_lens batch={seq_lens.shape[0]} does not match batch_size={batch_size}"
        )
    

    device = input_ids.device
    dtype = model.embed_tokens.weight.dtype

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads


    hidden_states = F.embedding(input_ids, model.embed_tokens.weight)

    position_ids = torch.tensor(
        [[position] for position in token_positions],
        device=device,
        dtype=torch.long
    )
    rope_x = torch.empty(
        batch_size,
        num_key_value_heads,
        1,
        head_dim,
        device=device,
        dtype=dtype
    )

    cos, sin = model.rotary_emb(rope_x, position_ids)

    for layer_id, layer in enumerate(model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        hidden_states = llama_decoder_layer_forward_batch_with_paged_attention(
            hidden_states=hidden_states,
            input_layernorm_weight=layer.input_layernorm.weight,
            post_attention_layernorm_weight=layer.post_attention_layernorm.weight,
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
            layer_id=layer_id,
            token_positions=token_positions,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            seq_lens=seq_lens,
            kv_cache_pool=kv_cache_pool,
            attention_backend=attention_backend,
            q_proj_bias=attn.q_proj.bias,
            k_proj_bias=attn.k_proj.bias,
            v_proj_bias=attn.v_proj.bias,
            o_proj_bias=attn.o_proj.bias,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
        )
    
    hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=model.norm.weight,
        eps=config.rms_norm_eps
    )

    logits = F.linear(hidden_states, hf_model.lm_head.weight, bias=None)
    return logits

