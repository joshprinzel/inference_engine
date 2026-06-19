from __future__ import annotations

import math

import torch
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    Repeate KV heads for grouped-query attention

    Args:
        hidden_states:
            Tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        num_key_value_groups:
            Number of query-head groups per KV head.
        
            For TinyLlama:
                num_q_heads = 32
                num_kv_heads = 4
                num_key_value_groups = 32 // 4 = 8

    Returns:
        Tensor of shape [batch, num_q_heads, seq_len, head_dim]
    """

    if num_key_value_groups == 1:
        return hidden_states
    
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape

    hidden_states = hidden_states[:,:,None,:,:]
    hidden_states = hidden_states.expand(
        batch_size,
        num_kv_heads,
        num_key_value_groups,
        seq_len,
        head_dim
    )
    return hidden_states.reshape(
        batch_size,
        num_kv_heads * num_key_value_groups,
        seq_len,
        head_dim
    )


def build_causal_mask(
        batch_size: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build additive causal mask for attention scores.

    Returned shape:
        [batch, l, q_len, kv_len]

    Values:
        0.0 for visible positions
        finfo(dtype).min for masked future positions
    """

    mask = torch.full(
        (q_len,kv_len),
        fill_value=torch.finfo(dtype).min,
        device=device,
        dtype=dtype
    )

    #For full-sequence prefill where q_len == kv_len:
    #token i can see keys <= i
    mask = torch.triu(mask,diagonal=1)
    
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.expand(batch_size,1,q_len,kv_len)


def llama_attention(
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        num_key_value_groups: int,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None
) -> torch.Tensor:
    """
    Llama grouped-query causal self-attention.

    Args:
        q:
            Query tensor of shape [batch, num_q_heads, q_len, head_dim]
        k:
            Key tensor of shape [batch, num_kv_heads, kv_len, head_dim]
        v:
            Value tensor of shape [batch, num_kv_heads, kv_len, head_dim]
        num_key_value_groups:
            num_q_heads // num_kv_heads
        attention_mask:
            Optional additive mask broadcastable to
            [batch, num_q_heads, q_len, kv_len].
    
    Returns:
        Attention output of shape [batch, num_q_heads, q_len, head_dim]
    """
    batch_size, num_q_heads, q_len, head_dim = q.shape
    _,_,kv_len,_ = k.shape

    if scaling is None:
        scaling = head_dim**-0.5

    k = repeat_kv(k, num_key_value_groups)
    v = repeat_kv(v, num_key_value_groups)

    assert k.shape == (batch_size, num_q_heads, kv_len, head_dim)
    assert v.shape == (batch_size, num_q_heads, kv_len, head_dim)

    attn_weights = torch.matmul(q,k.transpose(2,3)) * scaling
    if attention_mask is not None:
        attention_mask = attention_mask[:,:,:,:kv_len]
        attn_weights = attn_weights + attention_mask
    
    #HF Llama attention usually computes softmax in fp32, then recasts back
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    assert attn_output.shape == (batch_size, num_q_heads, q_len, head_dim)
    return attn_output


def merge_attention_heads(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Merge attention heads back into hidden dimension.

    Input:
        [batch, num_heads, seq_len, head_dim]
    
    Output:
        [batch, seq_len, num_heads * head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = attn_output.shape
    attn_output = attn_output.transpose(1,2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)

    return attn_output


def llama_attention_with_output_projection(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o_proj_weight: torch.Tensor,
        o_proj_bias: torch.Tensor | None,
        num_key_value_groups: int,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None
) -> torch.Tensor:
    """
    Full attention sublayer after Q/K/V/RoPE

        GQA attention -> merge heads -> o_proj
    
    Returns:
        [batch, seq_len, hidden_size]
    """

    attn_output = llama_attention(
        q=q,
        k=k,
        v=v,
        num_key_value_groups=num_key_value_groups,
        attention_mask=attention_mask,
        scaling=scaling
    )
    merged = merge_attention_heads(attn_output)
    return F.linear(merged, o_proj_weight, o_proj_bias)