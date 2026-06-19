from __future__ import annotations

import torch
import torch.nn.functional as F


def llama_swiglu_mlp(
        hidden_states: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        up_proj_weight: torch.Tensor,
        down_proj_weight: torch.Tensor,
        gate_proj_bias: torch.Tensor | None = None,
        up_proj_bias: torch.Tensor | None = None,
        down_proj_bias: torch.Tensor | None = None,        
) -> torch.Tensor:
    """
    Llama SwiGLU MLP.

    Formula:
        down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_states:
            [batch, seq_len, hidden_size]
        gate_proj_weight:
            [intermediate_size, hidden_size]
        up_proj_weight:
            [intermediate_size, hidden_size]
        down_proj_weight:
            [hidden_size, intermediate_size]

    Returns:
        [batch, seq_len, hidden_size]
    """

    gate = F.linear(hidden_states, gate_proj_weight, gate_proj_bias)
    up = F.linear(hidden_states, up_proj_weight, up_proj_bias)

    activated = F.silu(gate) * up
    return F.linear(activated, down_proj_weight, down_proj_bias)