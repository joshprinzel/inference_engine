from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from engines.llama.attention import build_decode_attention_mask
from engines.llama.cached_layer import llama_decoder_layer_forward_with_kv_cache
from engines.llama.rmsnorm import llama_rmsnorm

PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = list[PastKeyValue]

def call_hf_rotary_emb(
        rotary_emb: torch.nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    signature = inspect.signature(rotary_emb.forward)

    if "position_ids" in signature.parameters:
        return rotary_emb(x, position_ids)
    if "seq_len" in signature.parameters:
        seq_len = int(position_ids.shape[-1])
        return rotary_emb(x, seq_len=seq_len)
    return rotary_emb(x, position_ids)


def llama_model_forward_with_kv_cache_from_hf_weights(
        hf_model: torch.nn.Module,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
) -> tuple[torch.Tensor, PastKeyValues]:
    """
    Full Llama causal LM forward with append-only contiguous PyTorch K/V cache.

    This is still a correctness harness / transition implementation. It reads
    weights from the Hugging Face model object but does not call HF model.forward,
    decoder-layer forward, or attention forward.

    Args:
        hf_model:
            AutoModelForCausalLM instance used as a weight container.
        input_ids:
            [batch, q_len]
            During prefill, q_len is prompt length.
            During decode, q_len is usually 1.
        past_key_values:
            None for prefill, otherwise one (K, V) pair per decoder layer.

    Returns:
        logits:
            [batch, q_len, vocab_size]
        present_key_values:
            list of length num_hidden_layers.
            Each element:
                key:   [batch, kv_heads, past_len + q_len, head_dim]
                value: [batch, kv_heads, past_len + q_len, head_dim]
    """
    config = hf_model.config
    model = hf_model.model

    batch_size, q_len = input_ids.shape
    device = input_ids.device
    dtype = model.embed_tokens.weight.dtype

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    num_layers = config.num_hidden_layers

    if past_key_values is None:
        past_len = 0
        past_key_values = [(None,None) for _ in range(num_layers)]
    else:
        past_len = int(past_key_values[0][0].shape[2])
        assert len(past_key_values) == num_layers
    

    kv_len = past_len + q_len

    position_ids = torch.arange(
        past_len,
        past_len + q_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)

    hidden_states = F.embedding(input_ids, model.embed_tokens.weight)

    rope_x = torch.empty(
        batch_size,
        num_key_value_heads,
        q_len,
        head_dim,
        device=device,
        dtype=dtype
    )

    cos, sin = call_hf_rotary_emb(
        rotary_emb=model.rotary_emb,
        x=rope_x,
        position_ids=position_ids
    )

    attention_mask = build_decode_attention_mask(
        batch_size=batch_size,
        q_len=q_len,
        kv_len=kv_len,
        past_len=past_len,
        device=device,
        dtype=dtype
    )

    present_key_values: PastKeyValues = []

    for layer_index, layer in enumerate(model.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        past_key, past_value = past_key_values[layer_index]
        attention_scaling = getattr(attn, "scaling", head_dim**-0.5)

        hidden_states, present_key, present_value = llama_decoder_layer_forward_with_kv_cache(
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
            rms_norm_eps=config.rms_norm_eps,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            past_key=past_key,
            past_value=past_value,
            q_proj_bias=attn.q_proj.bias,
            k_proj_bias=attn.k_proj.bias,
            v_proj_bias=attn.v_proj.bias,
            o_proj_bias=attn.o_proj.bias,
            gate_proj_bias=mlp.gate_proj.bias,
            up_proj_bias=mlp.up_proj.bias,
            down_proj_bias=mlp.down_proj.bias,
            attention_scaling=attention_scaling,
        )

        present_key_values.append((present_key,present_value))

    hidden_states = llama_rmsnorm(
        hidden_states=hidden_states,
        weight=model.norm.weight,
        eps=config.rms_norm_eps
    )

    logits = F.linear(hidden_states, hf_model.lm_head.weight, bias=None)
    return logits, present_key_values