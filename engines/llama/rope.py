from __future__ import annotations

import torch

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Llama rotate_half convention.

    Splits the last dimension into two halves:

        x = [x1, x2]
    
    and returns:

        [-x2, x1]
    
    This matches Hugging Face Llama's rotate_half convention
    """
    half_dim = x.shape[-1] // 2

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    return torch.cat((-x2,x1), dim=-1)


def apply_llama_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Llama rotary positional embeddings to Q and K.

    Args:
        q:
            Query tensor of shape [batch, num_q_heads, seq_len, head_dim].
        k:
            Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        cos:
            Cosine tensor. Usually from HF rotary_emb with shape
            [batch, seq_len, head_dim].
        sin:
            Sine tensor. Usually from HF rotary_emb with shape
            [batch, seq_len, head_dim].
        unsqueeze_dim:
            Dimension used to broadcast cos/sin across heads.
            For q/k shaped [batch, heads, seq, head_dim], this should be 1.

    Returns:
        Tuple of:
            q_rot: [batch, num_q_heads, seq_len, head_dim]
            k_rot: [batch, num_kv_heads, seq_len, head_dim]
    """

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot
