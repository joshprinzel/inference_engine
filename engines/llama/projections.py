from __future__ import annotations

import torch
import torch.nn.functional as F

def llama_linear(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None

) -> torch.Tensor:
    """
    Apply a Llama linear projection using explicit weights.

    HF Llama projection weights are stored as:
        [out_features, in_features]
    
    For hidden_states:
        [batch, seq_len, in_features]
    
    the output is:
        [batch, seq_len, out_features]
    """

    return F.linear(hidden_states,weight,bias)


def project_qkv_with_weights(
        hidden_states: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
    q = llama_linear(hidden_states,q_weight,q_bias)
    k = llama_linear(hidden_states,k_weight,k_bias)
    v = llama_linear(hidden_states,v_weight,v_bias)
    return q,k,v


def reshape_qkv_for_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = q.shape

    q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
    k = k.view(batch_size, seq_len, num_key_value_heads, head_dim)
    v = v.view(batch_size, seq_len, num_key_value_heads,head_dim)

    q = q.transpose(1,2).contiguous()
    k = k.transpose(1,2).contiguous()
    v = v.transpose(1,2).contiguous()

    return q,k,v
