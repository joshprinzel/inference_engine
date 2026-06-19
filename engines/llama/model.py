from __future__ import annotations

import torch
import torch.nn.functional as F

from engines.llama.layer import llama_decoder_layer_forward
from engines.llama.rmsnorm import llama_rmsnorm


def llama_model_forward(
    input_ids: torch.Tensor,
    embed_tokens_weight: torch.Tensor,
    layers: list[object],
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor | None,
    rms_norm_eps: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Full Llama-style causal LM forward pass.

    This function intentionally takes HF layer/module objects for now as a
    weight source. It does not call their forward methods. It only reads weights.

    Later, this should be replaced by explicit LlamaWeights / LlamaLayerWeights
    containers.

    Args:
        input_ids:
            [batch, seq_len]
        embed_tokens_weight:
            [vocab_size, hidden_size]
        layers:
            List of HF LlamaDecoderLayer-like objects used as weight containers.
        final_norm_weight:
            [hidden_size]
        lm_head_weight:
            [vocab_size, hidden_size]
        cos/sin:
            RoPE tensors, usually [batch, seq_len, head_dim].
        attention_mask:
            Additive causal mask [batch, 1, seq_len, seq_len].

    Returns:
        logits:
            [batch, seq_len, vocab_size]
    """

    hidden_states = F.embedding(input_ids, embed_tokens_weight)

    for layer in layers:
        attn = layer.self_attn
        mlp = layer.mlp

        attention_scaling = getattr(attn, "scaling", head_dim**-0.5)

        hidden_states = llama_decoder_layer_forward(
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
            attention_mask=attention_mask,
            rms_norm_eps=rms_norm_eps,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            q_proj_bias=attn.q_proj.bias,
            k_proj_bias=attn.k_proj.bias,
            v_proj_bias=attn.v_proj.bias,
            o_proj_bias=attn.o_proj.bias,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
            attention_scaling=attention_scaling,
        )

    hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=final_norm_weight,
        eps=rms_norm_eps,
    )

    logits = F.linear(hidden_states, lm_head_weight, bias=None)

    return logits