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



# Cached Aware Attention Helpers Past This Point

def concat_past_key_value(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Append current-step K/V to previous cached K/V.

    Shapes:
        key_states: [batch, kv_heads, q_len, head_dim]
        value_states: [batch, kv_heads, q_len, head_dim]
        past_key: [batch, kv_heads, past_len, head_dim] or None
        past_value: [batch, kv_heads, past_len, head_dim] or None

    Returns:
        full_key:   [batch, kv_heads, past_len + q_len, head_dim]
        full_value: [batch, kv_heads, past_len + q_len, head_dim]
    """

    if past_key is None:
        full_key = key_states
    else:
        full_key = torch.cat([past_key, key_states], dim=2)
    

    if past_value is None:
        full_value = value_states
    else:
        full_value = torch.cat([past_value, value_states], dim=2)
    
    return full_key, full_value


def build_decode_attention_mask(
        batch_size: int,
        q_len: int,
        kv_len: int,
        past_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build additive causal mask for cached decode.

    For prefill:
        past_len = 0
        q_len = prompt length
        kv_len = prompt length

    For one-token decode:
        past_len = previous cache length
        q_len = 1
        kv_len = past_len + 1

    The rule is:
        query absolute position = past_len + query_index
        key absolute position = key_index
        mask if key_position > query_position
    """

    query_positions = past_len + torch.arange(q_len, device=device)
    key_positions = torch.arange(kv_len, device=device)

    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    mask = torch.zeros((q_len, kv_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)

    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.expand(batch_size,1,q_len, kv_len)

def llama_attention_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    num_key_value_groups: int,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GQA attention with append-only K/V cache.

    Args:
        q:
            [batch, q_heads, q_len, head_dim]
        k:
            [batch, kv_heads, q_len, head_dim]
        v:
            [batch, kv_heads, q_len, head_dim]
        past_key:
            [batch, kv_heads, past_len, head_dim] or None
        past_value:
            [batch, kv_heads, past_len, head_dim] or None

    Returns:
        attn_output:
            [batch, q_heads, q_len, head_dim]
        present_key:
            [batch, kv_heads, past_len + q_len, head_dim]
        present_value:
            [batch, kv_heads, past_len + q_len, head_dim]
    """

    batch_size, num_q_heads, q_len, head_dim = q.shape

    present_key, present_value = concat_past_key_value(
        key_states=k,
        value_states=v,
        past_key=past_key,
        past_value=past_value,
    )

    _, _, kv_len, _ = present_key.shape

    if scaling is None:
        scaling = head_dim**-0.5

    repeated_key = repeat_kv(present_key, num_key_value_groups)
    repeated_value = repeat_kv(present_value, num_key_value_groups)

    assert repeated_key.shape == (batch_size, num_q_heads, kv_len, head_dim)
    assert repeated_value.shape == (batch_size, num_q_heads, kv_len, head_dim)

    attn_weights = torch.matmul(q, repeated_key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, repeated_value)

    assert attn_output.shape == (batch_size, num_q_heads, q_len, head_dim)

    return attn_output, present_key, present_value


def llama_attention_with_kv_cache_and_output_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor | None,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    num_key_value_groups: int,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn_output, present_key, present_value = llama_attention_with_kv_cache(
        q=q,
        k=k,
        v=v,
        past_key=past_key,
        past_value=past_value,
        num_key_value_groups=num_key_value_groups,
        attention_mask=attention_mask,
        scaling=scaling,
    )

    merged = merge_attention_heads(attn_output)
    projected = F.linear(merged, o_proj_weight, o_proj_bias)

    return projected, present_key, present_value