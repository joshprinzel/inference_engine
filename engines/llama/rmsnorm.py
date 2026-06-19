from __future__ import annotations

import torch

def llama_rmsnorm(
        hidden_states: torch.tensor,
        weight: torch.tensor,
        eps: float
) -> torch.Tensor:
    """
    Llama RMSNorm.

    HF LlamaRMSNorm computes variance in float32 for numerical stability.
    then casts the normalized hidden states back to the input dtype before
    applying the learned weight.

    Args:
        hidden_states:
            Tensor of shape [..., hidden_size].
        weight:
            Learned RMSNorm weight of shape [hidden_size].
        eps:
            RMSNorm epsilon.
    """
    input_dtype = hidden_states.dtype

    hidden_states_fp32 = hidden_states.to(torch.float32)
    variance = hidden_states_fp32.pow(2).mean(dim=-1,keepdim=True)
    normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)

    #Recast back to input_dtype
    normalized = normalized.to(input_dtype)

    return weight * normalized