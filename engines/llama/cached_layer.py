from __future__ import annotations

import torch

from engines.llama.attention import llama_attention_with_kv_cache_and_output_projection
from engines.llama.mlp import llama_swiglu_mlp
from engines.llama.projections import project_qkv_with_weights, reshape_qkv_for_attention
from engines.llama.rmsnorm import llama_rmsnorm
from engines.llama.rope import apply_llama_rope


def llama_decoder_layer_forward_with_kv_cache(
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
    attention_mask: torch.Tensor | None,
    rms_norm_eps: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    q_proj_bias: torch.Tensor | None = None,
    k_proj_bias: torch.Tensor | None = None,
    v_proj_bias: torch.Tensor | None = None,
    o_proj_bias: torch.Tensor | None = None,
    gate_proj_bias: torch.Tensor | None = None,
    up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    attention_scaling: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decoder layer forward with append-only K/V cache.

    Returns:
        hidden_states:
            [batch, q_len, hidden_size]
        present_key:
            [batch, kv_heads, past_len + q_len, head_dim]
        present_value:
            [batch, kv_heads, past_len + q_len, head_dim]
    """

    num_key_value_groups = num_attention_heads // num_key_value_heads

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

    attn_out, present_key, present_value = llama_attention_with_kv_cache_and_output_projection(
        q=q_rot,
        k=k_rot,
        v=v_heads,
        o_proj_weight=o_proj_weight,
        o_proj_bias=o_proj_bias,
        past_key=past_key,
        past_value=past_value,
        num_key_value_groups=num_key_value_groups,
        attention_mask=attention_mask,
        scaling=attention_scaling,
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

    return hidden_states, present_key, present_value