from __future__ import annotations

import torch
import torch.nn.functional as F

from engines.llama.attention import merge_attention_heads
from engines.llama.mlp import llama_swiglu_mlp
from engines.llama.projections import project_qkv_with_weights, reshape_qkv_for_attention
from engines.llama.rmsnorm import llama_rmsnorm
from engines.llama.rope import apply_llama_rope
from runtime.attention_backend import AttentionBackend
from runtime.kv_cache_pool import KVCachePool

def llama_decoder_layer_forward_with_paged_attention(
    hidden_states: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    layer_id: int,
    token_position: int,
    block_table: list[int],
    block_tables_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_pool: KVCachePool,
    attention_backend: AttentionBackend,
    q_proj_bias: torch.Tensor | None = None,
    k_proj_bias: torch.Tensor | None = None,
    v_proj_bias: torch.Tensor | None = None,
    o_proj_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    One-token Llama decoder layer using paged attention

    Assumptions:
        - hidden_states shape is [batch=1. q_len=1, hidden_size]
        - KVCachePool already contains previous K/V for this request
        - this function writes the current-token K/V into KVCachePool
        - attention_backend.decode reads K/V from KVCachePool using block tables
        - attention_backend.decode returns [batch, num_attention_heads, head_dim]

    Returns:
        hidden_state shape [1,1,hidden_size]
    """

    batch_size, q_len, _ = hidden_states.shape

    if batch_size != 1:
        raise ValueError(
            f"paged decoder layer currently expects batch_size=1, got {batch_size}"
        )
    if q_len != 1:
        raise ValueError(
            f"paged decoder layer currently expects q_len=1, got {q_len}"
        )
    
    residual = hidden_states

    normed_hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=input_layernorm_weight,
        eps=rms_norm_eps
    )

    q,k,v = project_qkv_with_weights(
        hidden_states=normed_hidden_states,
        q_weight=q_proj_weight,
        k_weight=k_proj_weight,
        v_weight=v_proj_weight,
        q_bias=q_proj_bias,
        k_bias=k_proj_bias,
        v_bias=v_proj_bias
    )

    q_heads, k_heads, v_heads = reshape_qkv_for_attention(
        q=q,
        k=k,
        v=v,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim
    )

    q_rot, k_rot = apply_llama_rope(
        q=q_heads,
        k=k_heads,
        cos=cos,
        sin=sin,
        unsqueeze_dim=1
    )

    # Write the current token K/V before attention
    # Decode attention should attend over previous tokens + current token
    kv_cache_pool.write_request_token(
        layer_id=layer_id,
        block_table=block_table,
        token_position=token_position,
        key=k_rot[0, :, 0, :].contiguous(),
        value=v_heads[0, :, 0, :].contiguous()
    )

    attn_heads = attention_backend.decode(
        q=q_rot[:,:,0,:].contiguous(),
        cache_pool=kv_cache_pool,
        layer_id=layer_id,
        block_tables=block_tables_tensor,
        seq_lens=seq_lens
    )

    #[batch, heads, head_dim] -> [batch, heads, 1, head_dim]
    attn_heads = attn_heads.unsqueeze(2)

    merged_attn = merge_attention_heads(attn_heads)

    attn_out = F.linear(
        merged_attn,
        o_proj_weight,
        o_proj_bias
    )

    hidden_states = residual + attn_out
    residual = hidden_states

    normed_hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=post_attention_layernorm_weight,
        eps=rms_norm_eps
    )

    mlp_out = llama_swiglu_mlp(
        hidden_states=normed_hidden_states,
        gate_proj_weight=gate_proj_weight,
        up_proj_weight=up_proj_weight,
        down_proj_weight=down_proj_weight,
        gate_proj_bias=gate_proj_bias,
        up_proj_bias=up_proj_bias,
        down_proj_bias=down_proj_bias
    )

    hidden_states = residual + mlp_out
    return hidden_states


def llama_decoder_layer_forward_batch_with_paged_attention(
    hidden_states: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    layer_id: int,
    token_positions: list[int],
    block_tables: list[list[int]],
    block_tables_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_pool: KVCachePool,
    attention_backend: AttentionBackend,
    q_proj_bias: torch.Tensor | None = None,
    k_proj_bias: torch.Tensor | None = None,
    v_proj_bias: torch.Tensor | None = None,
    o_proj_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Batched one-token Llama decoder layer using paged attention.

    hidden_states: [B, 1, hidden_size]
    attention_backend.decode returns: [B, num_attention_heads, head_dim]
    """

    batch_size, q_len, _ = hidden_states.shape

    if q_len != 1:
        raise ValueError(
            f"paged decoder batch layer expects q_len=1, got {q_len}"
        )

    if batch_size != len(token_positions):
        raise ValueError(
            f"batch_size={batch_size} does not match "
            f"len(token_positions)={len(token_positions)}"
        )

    if batch_size != len(block_tables):
        raise ValueError(
            f"batch_size={batch_size} does not match "
            f"len(block_tables)={len(block_tables)}"
        )

    residual = hidden_states

    normed_hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=input_layernorm_weight,
        eps=rms_norm_eps,
    )

    q, k, v = project_qkv_with_weights(
        hidden_states=normed_hidden_states,
        q_weight=q_proj_weight,
        k_weight=k_proj_weight,
        v_weight=v_proj_weight,
        q_bias=q_proj_bias,
        k_bias=k_proj_bias,
        v_bias=v_proj_bias,
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

    for batch_index in range(batch_size):
        kv_cache_pool.write_request_token(
            layer_id=layer_id,
            block_table=block_tables[batch_index],
            token_position=token_positions[batch_index],
            key=k_rot[batch_index, :, 0, :].contiguous(),
            value=v_heads[batch_index, :, 0, :].contiguous(),
        )

    attn_heads = attention_backend.decode(
        q=q_rot[:, :, 0, :].contiguous(),
        cache_pool=kv_cache_pool,
        layer_id=layer_id,
        block_tables=block_tables_tensor,
        seq_lens=seq_lens,
    )

    attn_heads = attn_heads.unsqueeze(2)

    merged_attn = merge_attention_heads(attn_heads)

    attn_out = F.linear(
        merged_attn,
        o_proj_weight,
        o_proj_bias,
    )

    hidden_states = residual + attn_out

    residual = hidden_states

    normed_hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=post_attention_layernorm_weight,
        eps=rms_norm_eps,
    )

    mlp_out = llama_swiglu_mlp(
        hidden_states=normed_hidden_states,
        gate_proj_weight=gate_proj_weight,
        up_proj_weight=up_proj_weight,
        down_proj_weight=down_proj_weight,
        gate_proj_bias=gate_proj_bias,
        up_proj_bias=up_proj_bias,
        down_proj_bias=down_proj_bias,
    )

    hidden_states = residual + mlp_out

    return hidden_states

