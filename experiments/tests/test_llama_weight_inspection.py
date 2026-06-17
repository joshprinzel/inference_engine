"""
Inspect a Llama-compatible Hugging Face checkpoint before implementing
CustomLlamaDecodeEngine.

This is intentionally an experiment/inspection script, not core runtime logic.

The goal is to anchor the custom engine in the actual model config, module names,
weight shapes, tokenizer behavior, and KV-cache shape before writing any custom
Llama engine code.

Recommended real target:
    TinyLlama/TinyLlama-1.1B-Chat-v1.0

Fast debug fallback:
    hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import pytest


DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEBUG_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@dataclass(frozen=True)
class WeightSpec:
    label: str
    name: str


EXPECTED_FIRST_LAYER_WEIGHTS: tuple[WeightSpec, ...] = (
    WeightSpec("embed_tokens", "model.embed_tokens.weight"),
    WeightSpec("q_proj", "model.layers.0.self_attn.q_proj.weight"),
    WeightSpec("k_proj", "model.layers.0.self_attn.k_proj.weight"),
    WeightSpec("v_proj", "model.layers.0.self_attn.v_proj.weight"),
    WeightSpec("o_proj", "model.layers.0.self_attn.o_proj.weight"),
    WeightSpec("gate_proj", "model.layers.0.mlp.gate_proj.weight"),
    WeightSpec("up_proj", "model.layers.0.mlp.up_proj.weight"),
    WeightSpec("down_proj", "model.layers.0.mlp.down_proj.weight"),
    WeightSpec("input_layernorm", "model.layers.0.input_layernorm.weight"),
    WeightSpec(
        "post_attention_layernorm",
        "model.layers.0.post_attention_layernorm.weight",
    ),
    WeightSpec("final_norm", "model.norm.weight"),
    WeightSpec("lm_head", "lm_head.weight"),
)


def get_attr(config: object, name: str, default: object = None) -> object:
    return getattr(config, name, default)


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def resolve_rope_theta(config: object) -> object:
    """
    TinyLlama may expose rope_theta through rope_scaling instead of config.rope_theta.

    We do not normalize this into runtime config yet. This script only reports what
    the checkpoint exposes.
    """

    rope_theta = get_attr(config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta

    rope_scaling = get_attr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        return rope_scaling.get("rope_theta", None)

    return None


def print_config_summary(config: object) -> None:
    hidden_size = get_attr(config, "hidden_size")
    num_attention_heads = get_attr(config, "num_attention_heads")
    num_key_value_heads = get_attr(config, "num_key_value_heads", num_attention_heads)

    if hidden_size is not None and num_attention_heads is not None:
        head_dim = hidden_size // num_attention_heads
    else:
        head_dim = get_attr(config, "head_dim", None)

    if num_attention_heads and num_key_value_heads:
        num_query_groups = num_attention_heads // num_key_value_heads
    else:
        num_query_groups = None

    print_section("CONFIG SUMMARY")

    fields = {
        "model_type": get_attr(config, "model_type"),
        "architectures": get_attr(config, "architectures"),
        "hidden_size": hidden_size,
        "intermediate_size": get_attr(config, "intermediate_size"),
        "num_hidden_layers": get_attr(config, "num_hidden_layers"),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "num_query_groups": num_query_groups,
        "max_position_embeddings": get_attr(config, "max_position_embeddings"),
        "rope_theta": get_attr(config, "rope_theta", None),
        "resolved_rope_theta": resolve_rope_theta(config),
        "rope_scaling": get_attr(config, "rope_scaling", None),
        "rms_norm_eps": get_attr(config, "rms_norm_eps"),
        "hidden_act": get_attr(config, "hidden_act"),
        "vocab_size": get_attr(config, "vocab_size"),
        "tie_word_embeddings": get_attr(config, "tie_word_embeddings"),
        "torch_dtype": get_attr(config, "torch_dtype"),
        "use_cache": get_attr(config, "use_cache"),
    }

    for key, value in fields.items():
        print(f"{key:28s}: {value}")


def print_tokenizer_summary(tokenizer: object) -> None:
    print_section("TOKENIZER SUMMARY")

    fields = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }

    for key, value in fields.items():
        print(f"{key:28s}: {value}")


def print_named_module_summary(model: torch.nn.Module, max_lines: int = 80) -> None:
    print_section("FIRST NAMED MODULES")

    for idx, (name, module) in enumerate(model.named_modules()):
        if idx >= max_lines:
            print(f"... truncated after {max_lines} modules")
            return

        print(f"{idx:04d}  {name:60s}  {module.__class__.__name__}")


def print_weight_shapes(model: torch.nn.Module) -> None:
    print_section("EXPECTED WEIGHT SHAPES")

    named_parameters = dict(model.named_parameters())

    for spec in EXPECTED_FIRST_LAYER_WEIGHTS:
        param = named_parameters.get(spec.name)

        if param is None:
            print(f"{spec.label:28s}: MISSING expected={spec.name}")
            continue

        shape = tuple(param.shape)
        dtype = str(param.dtype).replace("torch.", "")
        device = str(param.device)
        requires_grad = param.requires_grad

        print(
            f"{spec.label:28s}: "
            f"shape={shape!s:24s} "
            f"dtype={dtype:10s} "
            f"device={device:8s} "
            f"requires_grad={requires_grad}"
        )


def print_first_layer_parameter_names(
    model: torch.nn.Module,
    prefixes: Iterable[str] = (
        "model.embed_tokens",
        "model.layers.0",
        "model.norm",
        "lm_head",
    ),
) -> None:
    print_section("FIRST LAYER PARAMETER NAMES")

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            print(f"{name:72s} {tuple(param.shape)}")


def extract_layer_kv_from_cache(
    past_key_values: object,
    layer_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract K/V tensors from Hugging Face cache variants.

    Supported cases:
      1. Legacy tuple/list cache:
            past_key_values[layer_idx] -> (key, value)

      2. DynamicCache-style object with:
            past_key_values.key_cache[layer_idx]
            past_key_values.value_cache[layer_idx]

      3. Newer per-layer cache containers:
            past_key_values.layers[layer_idx].keys / values
         or similar attribute names.

    This is inspection-only glue. Do not design the real runtime around HF cache
    internals. CustomLlamaDecodeEngine should own its own KVCachePool path.
    """

    if isinstance(past_key_values, (tuple, list)):
        return past_key_values[layer_idx]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = getattr(past_key_values, "key_cache")
        value_cache = getattr(past_key_values, "value_cache")
        return key_cache[layer_idx], value_cache[layer_idx]

    if hasattr(past_key_values, "layers"):
        layers = getattr(past_key_values, "layers")
        layer = layers[layer_idx]

        key_attr_candidates = (
            "keys",
            "key",
            "k",
            "key_states",
            "key_cache",
        )
        value_attr_candidates = (
            "values",
            "value",
            "v",
            "value_states",
            "value_cache",
        )

        key = None
        value = None

        for attr_name in key_attr_candidates:
            if hasattr(layer, attr_name):
                key = getattr(layer, attr_name)
                break

        for attr_name in value_attr_candidates:
            if hasattr(layer, attr_name):
                value = getattr(layer, attr_name)
                break

        if key is not None and value is not None:
            return key, value

    public_attrs = [
        name for name in dir(past_key_values)
        if not name.startswith("_")
    ]

    raise TypeError(
        "Could not extract K/V tensors from past_key_values. "
        f"type={type(past_key_values).__name__}, "
        f"public_attrs={public_attrs}"
    )


def print_cache_summary(past_key_values: object) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"past_key_values type: {type(past_key_values).__name__}")

    if hasattr(past_key_values, "__len__"):
        try:
            print(f"num past_key_values layers: {len(past_key_values)}")
        except TypeError:
            print("num past_key_values layers: len() unsupported")
    elif hasattr(past_key_values, "key_cache"):
        key_cache = getattr(past_key_values, "key_cache")
        print(f"num past_key_values layers: {len(key_cache)}")
    elif hasattr(past_key_values, "layers"):
        layers = getattr(past_key_values, "layers")
        print(f"num past_key_values layers: {len(layers)}")
    else:
        print("num past_key_values layers: unknown")

    first_k, first_v = extract_layer_kv_from_cache(past_key_values, layer_idx=0)

    print(f"layer 0 past K shape: {tuple(first_k.shape)}")
    print(f"layer 0 past V shape: {tuple(first_v.shape)}")
    print(f"layer 0 past K dtype: {first_k.dtype}")
    print(f"layer 0 past V dtype: {first_v.dtype}")
    print(f"layer 0 past K device: {first_k.device}")
    print(f"layer 0 past V device: {first_v.device}")

    return first_k, first_v


def validate_expected_llama_shapes(
    config: object,
    first_k: torch.Tensor,
    first_v: torch.Tensor,
    input_seq_len: int,
) -> None:
    print_section("SHAPE VALIDATION")

    hidden_size = get_attr(config, "hidden_size")
    num_attention_heads = get_attr(config, "num_attention_heads")
    num_key_value_heads = get_attr(config, "num_key_value_heads", num_attention_heads)

    if hidden_size is None or num_attention_heads is None or num_key_value_heads is None:
        print("Skipping validation: missing config fields.")
        return

    expected_head_dim = hidden_size // num_attention_heads

    expected_cache_shape = (
        1,
        num_key_value_heads,
        input_seq_len,
        expected_head_dim,
    )

    print(f"expected cache shape: {expected_cache_shape}")
    print(f"actual K shape:        {tuple(first_k.shape)}")
    print(f"actual V shape:        {tuple(first_v.shape)}")

    if tuple(first_k.shape) != expected_cache_shape:
        print("WARNING: K cache shape does not match expected Llama GQA layout.")
    else:
        print("K cache shape matches expected Llama GQA layout.")

    if tuple(first_v.shape) != expected_cache_shape:
        print("WARNING: V cache shape does not match expected Llama GQA layout.")
    else:
        print("V cache shape matches expected Llama GQA layout.")


def run_tiny_forward_smoke(
    model: torch.nn.Module,
    tokenizer: object,
    config: object,
    prompt: str,
    device: str,
) -> None:
    print_section("FORWARD SMOKE TEST")

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    print(f"prompt: {prompt!r}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"input_ids: {input_ids.tolist()}")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)

    logits = outputs.logits
    past_key_values = outputs.past_key_values

    print(f"logits shape: {tuple(logits.shape)}")
    print(f"logits dtype: {logits.dtype}")

    first_k, first_v = print_cache_summary(past_key_values)

    input_seq_len = input_ids.shape[1]
    validate_expected_llama_shapes(
        config=config,
        first_k=first_k,
        first_v=first_v,
        input_seq_len=input_seq_len,
    )

    next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    next_text = tokenizer.decode([next_token_id])

    print()
    print(f"greedy next_token_id: {next_token_id}")
    print(f"greedy next_text: {next_text!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Hugging Face model id. Use TinyLlama for real inspection or "
            "hf-internal-testing/tiny-random-LlamaForCausalLM for fast debug."
        ),
    )
    parser.add_argument(
        "--debug-model",
        action="store_true",
        help=f"Use {DEBUG_MODEL_ID}.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Not expected for TinyLlama. Leave false unless a checkpoint requires it.",
    )
    parser.add_argument(
        "--print-modules",
        action="store_true",
        help="Print first named modules. Useful during first inspection.",
    )

    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main() -> None:
    args = parse_args()

    if args.debug_model:
        args.model_id = DEBUG_MODEL_ID

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")

    torch_dtype = resolve_dtype(args.dtype)

    print_section("LOAD")
    print(f"model_id: {args.model_id}")
    print(f"device:   {args.device}")
    print(f"dtype:    {torch_dtype}")

    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    model.to(args.device)
    model.eval()

    print_config_summary(config)
    print_tokenizer_summary(tokenizer)

    if args.print_modules:
        print_named_module_summary(model)

    print_weight_shapes(model)
    print_first_layer_parameter_names(model)

    run_tiny_forward_smoke(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        device=args.device,
    )



@pytest.mark.slow
def test_llama_weight_inspection() -> None:
    main()


if __name__ == "__main__":
    main()