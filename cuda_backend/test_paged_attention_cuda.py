import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from paged_attention_reference import paged_attention_decode_reference

import paged_attention_cuda


KERNEL_NAME = "paged_attention_cuda_v2"
SEQ_LENS = [1, 7, 8, 9, 19]
MAX_ALLOWED_DIFF = 1e-2


def fill_request_kv(
    cache_pool: KVCachePool,
    layer_id: int,
    block_table: list[int],
    seq_len: int,
) -> None:
    for token_position in range(seq_len):
        key = torch.randn(
            cache_pool.layout.num_kv_heads,
            cache_pool.layout.head_dim,
            device=cache_pool.key_cache.device,
            dtype=cache_pool.key_cache.dtype,
        )

        value = torch.randn(
            cache_pool.layout.num_kv_heads,
            cache_pool.layout.head_dim,
            device=cache_pool.value_cache.device,
            dtype=cache_pool.value_cache.dtype,
        )

        cache_pool.write_request_token(
            layer_id=layer_id,
            block_table=block_table,
            token_position=token_position,
            key=key,
            value=value,
        )


def run_one_case(
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    layout = KVCacheLayout(
        num_layers=2,
        total_blocks=16,
        block_size_tokens=8,
        num_kv_heads=2,
        head_dim=32,
        dtype="float16",
        device="cuda",
    )

    block_manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    layer_id = 0
    num_query_heads = layout.num_kv_heads

    request_id = f"req-seq-{seq_len}"
    block_table = block_manager.allocate_for_tokens(
        request_id=request_id,
        num_tokens=seq_len,
    )

    fill_request_kv(
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    q = torch.randn(
        num_query_heads,
        layout.head_dim,
        device="cuda",
        dtype=layout.torch_dtype,
    )

    block_table_tensor = torch.tensor(
        block_table,
        dtype=torch.int32,
        device="cuda",
    ).contiguous()

    reference_output = paged_attention_decode_reference(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len,
    )

    cuda_output = paged_attention_cuda.paged_attention_decode(
        q.contiguous(),
        cache_pool.key_cache.contiguous(),
        cache_pool.value_cache.contiguous(),
        block_table_tensor,
        layer_id,
        seq_len,
    )

    diff = cuda_output.float() - reference_output.float()
    max_abs_diff = diff.abs().max().item()

    reference_finite = torch.isfinite(reference_output).all().item()
    cuda_finite = torch.isfinite(cuda_output).all().item()
    diff_finite = torch.isfinite(diff).all().item()

    passed = (
        reference_finite
        and cuda_finite
        and diff_finite
        and max_abs_diff < MAX_ALLOWED_DIFF
    )

    return {
        "kernel_name": KERNEL_NAME,
        "seq_len": seq_len,
        "seed": seed,
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "reference_finite": bool(reference_finite),
        "cuda_finite": bool(cuda_finite),
        "diff_finite": bool(diff_finite),
        "layout": layout.snapshot(),
        "block_table": block_table,
        "reference_output_shape": tuple(reference_output.shape),
        "cuda_output_shape": tuple(cuda_output.shape),
        "cuda_output_sample": cuda_output.flatten()[:8].detach().cpu().tolist(),
        "reference_output_sample": reference_output.flatten()[:8].detach().cpu().tolist(),
        "diff_sample": diff.flatten()[:8].detach().cpu().tolist(),
    }


def write_json_report(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "kernel_name": KERNEL_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "loaded_extension": str(paged_attention_cuda.__file__),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "max_allowed_diff": MAX_ALLOWED_DIFF,
        "all_passed": all(result["passed"] for result in results),
        "results": results,
    }

    output_path.write_text(json.dumps(payload, indent=2))


def write_markdown_report(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_passed = all(result["passed"] for result in results)
    first_layout = results[0]["layout"] if results else {}

    lines = []

    lines.append(f"# {KERNEL_NAME} Validation")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Validate that the CUDA C++ paged attention decode kernel matches the "
        "PyTorch paged attention reference across important sequence-length cases."
    )
    lines.append("")
    lines.append("This test is correctness-focused, not performance-focused.")
    lines.append("")
    lines.append("## Kernel Scope")
    lines.append("")
    lines.append("- Decode-only attention")
    lines.append("- Single request")
    lines.append("- One CUDA block per attention head")
    lines.append("- `num_query_heads == num_kv_heads`")
    lines.append("- FP16 inputs with FP32 accumulation")
    lines.append("- Paged KV layout using physical block tables")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Loaded extension: `{paged_attention_cuda.__file__}`")
    lines.append(f"- PyTorch: `{torch.__version__}`")
    lines.append(f"- CUDA: `{torch.version.cuda}`")
    lines.append(f"- Device: `{torch.cuda.get_device_name(0)}`")
    lines.append("")
    lines.append("## Layout")
    lines.append("")
    lines.append(f"- num_layers: `{first_layout.get('num_layers')}`")
    lines.append(f"- total_blocks: `{first_layout.get('total_blocks')}`")
    lines.append(f"- block_size_tokens: `{first_layout.get('block_size_tokens')}`")
    lines.append(f"- num_kv_heads: `{first_layout.get('num_kv_heads')}`")
    lines.append(f"- head_dim: `{first_layout.get('head_dim')}`")
    lines.append(f"- dtype: `{first_layout.get('dtype')}`")
    lines.append("")
    lines.append("## Correctness Grid")
    lines.append("")
    lines.append("| seq_len | block_table | max_abs_diff | passed |")
    lines.append("|---:|---|---:|---|")

    for result in results:
        lines.append(
            f"| {result['seq_len']} | `{result['block_table']}` | "
            f"{result['max_abs_diff']:.8f} | {result['passed']} |"
        )

    lines.append("")
    lines.append("## Result")
    lines.append("")
    if all_passed:
        lines.append("All correctness cases passed.")
    else:
        lines.append("One or more correctness cases failed.")
    lines.append("")
    lines.append("## Why These Sequence Lengths Matter")
    lines.append("")
    lines.append("- `1`: one-token attention; output should equal the only value vector")
    lines.append("- `7`: partial first block")
    lines.append("- `8`: exact block boundary")
    lines.append("- `9`: crosses from block 0 into block 1")
    lines.append("- `19`: multi-block sequence")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The CUDA kernel consumes the same ABI as the Python reference: query tensor, "
        "physical key/value cache tensors, block table, layer id, and sequence length."
    )
    lines.append("")
    lines.append(
        "Passing this grid validates that the kernel can walk the paged KV block table "
        "and reproduce dense/reference attention results across block-boundary cases."
    )
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append(
        "If this is v2, the next step is either benchmarking against v1 or moving toward "
        "a more efficient kernel that avoids repeated QK recomputation."
    )
    lines.append("")

    output_path.write_text("\n".join(lines))


def print_terminal_summary(results: list[dict[str, Any]]) -> None:
    print(f"{KERNEL_NAME}")
    print("---")
    print("loaded extension:", paged_attention_cuda.__file__)
    print("seq_len | block_table | max_abs_diff | passed")
    print("--- | --- | --- | ---")

    for result in results:
        print(
            f"{result['seq_len']} | "
            f"{result['block_table']} | "
            f"{result['max_abs_diff']:.8f} | "
            f"{result['passed']}"
        )

    print()
    print("all_passed:", all(result["passed"] for result in results))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    results = []

    for index, seq_len in enumerate(SEQ_LENS):
        result = run_one_case(
            seq_len=seq_len,
            seed=index,
        )
        results.append(result)

    print_terminal_summary(results)

    results_json_path = Path("results") / f"{KERNEL_NAME}_results.json"
    markdown_path = (
        PROJECT_ROOT
        / "docs"
        / "kernel_iterations"
        / f"{KERNEL_NAME}.md"
    )

    write_json_report(
        results=results,
        output_path=results_json_path,
    )

    write_markdown_report(
        results=results,
        output_path=markdown_path,
    )

    print()
    print(f"wrote JSON: {results_json_path}")
    print(f"wrote markdown: {markdown_path}")

    assert all(result["passed"] for result in results)


if __name__ == "__main__":
    main()