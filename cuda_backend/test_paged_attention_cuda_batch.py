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


KERNEL_NAME = "paged_attention_cuda_v5_batched"
SEQ_LEN_CASES = [
    [1],
    [7, 8],
    [9, 19, 32, 64],
    [64, 128, 256, 512],
]
MAX_ALLOWED_DIFF = 1e-2

NUM_LAYERS = 2
TOTAL_BLOCKS = 4096
BLOCK_SIZE_TOKENS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 32
DTYPE = "float16"
DEVICE = "cuda"


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


def make_padded_block_tables(
    block_tables: list[list[int]],
    pad_value: int = -1,
) -> torch.Tensor:
    max_blocks = max(len(block_table) for block_table in block_tables)

    padded = []

    for block_table in block_tables:
        row = block_table + [pad_value] * (max_blocks - len(block_table))
        padded.append(row)

    return torch.tensor(
        padded,
        dtype=torch.int32,
        device=DEVICE,
    ).contiguous()


def make_case(
    seq_lens: list[int],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    layout = KVCacheLayout(
        num_layers=NUM_LAYERS,
        total_blocks=TOTAL_BLOCKS,
        block_size_tokens=BLOCK_SIZE_TOKENS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )

    block_manager = KVBlockManager(
        total_blocks=layout.total_blocks,
        block_size_tokens=layout.block_size_tokens,
    )

    cache_pool = KVCachePool(layout)
    cache_pool.zero_()

    layer_id = 0
    block_tables: list[list[int]] = []

    for request_index, seq_len in enumerate(seq_lens):
        request_id = f"batch-req-{seed}-{request_index}"

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

        block_tables.append(block_table)

    batch_size = len(seq_lens)

    q = torch.randn(
        batch_size,
        layout.num_kv_heads,
        layout.head_dim,
        device=DEVICE,
        dtype=layout.torch_dtype,
    ).contiguous()

    block_tables_tensor = make_padded_block_tables(block_tables)

    seq_lens_tensor = torch.tensor(
        seq_lens,
        dtype=torch.int32,
        device=DEVICE,
    ).contiguous()

    return {
        "layout": layout,
        "cache_pool": cache_pool,
        "layer_id": layer_id,
        "seq_lens": seq_lens,
        "seq_lens_tensor": seq_lens_tensor,
        "block_tables": block_tables,
        "block_tables_tensor": block_tables_tensor,
        "q": q,
    }


def reference_batch_decode(
    q: torch.Tensor,
    cache_pool: KVCachePool,
    layer_id: int,
    block_tables: list[list[int]],
    seq_lens: list[int],
) -> torch.Tensor:
    outputs = []

    for batch_index, seq_len in enumerate(seq_lens):
        output_i = paged_attention_decode_reference(
            q=q[batch_index],
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_table=block_tables[batch_index],
            seq_len=seq_len,
        )

        outputs.append(output_i)

    return torch.stack(outputs, dim=0)


def run_one_case(
    seq_lens: list[int],
    seed: int,
) -> dict[str, Any]:
    case = make_case(
        seq_lens=seq_lens,
        seed=seed,
    )

    layout: KVCacheLayout = case["layout"]
    cache_pool: KVCachePool = case["cache_pool"]
    layer_id: int = case["layer_id"]
    block_tables: list[list[int]] = case["block_tables"]
    block_tables_tensor: torch.Tensor = case["block_tables_tensor"]
    seq_lens_tensor: torch.Tensor = case["seq_lens_tensor"]
    q: torch.Tensor = case["q"]

    reference_output = reference_batch_decode(
        q=q,
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )

    cuda_output = paged_attention_cuda.paged_attention_decode_batch(
        q,
        cache_pool.key_cache,
        cache_pool.value_cache,
        block_tables_tensor,
        seq_lens_tensor,
        layer_id,
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
        "seed": seed,
        "batch_size": len(seq_lens),
        "seq_lens": seq_lens,
        "passed": bool(passed),
        "max_abs_diff": max_abs_diff,
        "reference_finite": bool(reference_finite),
        "cuda_finite": bool(cuda_finite),
        "diff_finite": bool(diff_finite),
        "layout": layout.snapshot(),
        "block_tables": block_tables,
        "block_tables_shape": tuple(block_tables_tensor.shape),
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
        "Validate the batched CUDA paged attention decode kernel against the "
        "single-request PyTorch paged attention reference."
    )
    lines.append("")
    lines.append("## Kernel Scope")
    lines.append("")
    lines.append("- Decode-only attention")
    lines.append("- Batched requests")
    lines.append("- Variable sequence lengths per batch")
    lines.append("- One CUDA block per sequence/head pair")
    lines.append("- `num_query_heads == num_kv_heads`")
    lines.append("- FP16 inputs with FP32 accumulation")
    lines.append("- Paged KV layout using padded physical block tables")
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
    lines.append("| batch_size | seq_lens | block_tables_shape | max_abs_diff | passed |")
    lines.append("|---:|---|---|---:|---|")

    for result in results:
        lines.append(
            f"| {result['batch_size']} "
            f"| `{result['seq_lens']}` "
            f"| `{result['block_tables_shape']}` "
            f"| {result['max_abs_diff']:.8f} "
            f"| {result['passed']} |"
        )

    lines.append("")
    lines.append("## Result")
    lines.append("")
    if all_passed:
        lines.append("All batched correctness cases passed.")
    else:
        lines.append("One or more batched correctness cases failed.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Passing this grid validates that the CUDA kernel can map each batch row "
        "to its own padded block table, use its own sequence length, and produce "
        "the same output as independently decoding each request with the reference path."
    )
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append(
        "The next step is a batch-size sweep benchmark to verify that more "
        "sequence/head CTAs improve GPU occupancy and serving-shaped throughput."
    )
    lines.append("")

    output_path.write_text("\n".join(lines))


def print_terminal_summary(results: list[dict[str, Any]]) -> None:
    print(KERNEL_NAME)
    print("---")
    print("loaded extension:", paged_attention_cuda.__file__)
    print("batch_size | seq_lens | block_tables_shape | max_abs_diff | passed")
    print("--- | --- | --- | --- | ---")

    for result in results:
        print(
            f"{result['batch_size']} | "
            f"{result['seq_lens']} | "
            f"{result['block_tables_shape']} | "
            f"{result['max_abs_diff']:.8f} | "
            f"{result['passed']}"
        )

    print()
    print("all_passed:", all(result["passed"] for result in results))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    results = []

    for index, seq_lens in enumerate(SEQ_LEN_CASES):
        result = run_one_case(
            seq_lens=seq_lens,
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
    print(f"wrote JSON: {results_json_path.resolve()}")
    print(f"wrote markdown: {markdown_path.resolve()}")

    assert all(result["passed"] for result in results)


if __name__ == "__main__":
    main()