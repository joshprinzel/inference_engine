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
BENCH_NAME = f"{KERNEL_NAME}_bench"

SEQ_LENS = [1, 8, 19, 64, 128, 256]
BLOCK_SIZE_TOKENS = 8
NUM_LAYERS = 2
TOTAL_BLOCKS = 512
NUM_KV_HEADS = 2
HEAD_DIM = 32
DTYPE = "float16"
DEVICE = "cuda"

WARMUP_ITERS = 50
BENCH_ITERS = 500


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


def make_case(
    seq_len: int,
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
    request_id = f"bench-req-seq-{seq_len}"

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
        layout.num_kv_heads,
        layout.head_dim,
        device=DEVICE,
        dtype=layout.torch_dtype,
    ).contiguous()

    block_table_tensor = torch.tensor(
        block_table,
        dtype=torch.int32,
        device=DEVICE,
    ).contiguous()

    return {
        "layout": layout,
        "cache_pool": cache_pool,
        "layer_id": layer_id,
        "seq_len": seq_len,
        "block_table": block_table,
        "block_table_tensor": block_table_tensor,
        "q": q,
    }


def benchmark_cuda_events(
    fn,
    warmup_iters: int,
    bench_iters: int,
) -> float:
    for _ in range(warmup_iters):
        fn()

    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()

    for _ in range(bench_iters):
        fn()

    ender.record()
    torch.cuda.synchronize()

    total_ms = starter.elapsed_time(ender)
    return total_ms / bench_iters


def run_one_case(
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    case = make_case(seq_len=seq_len, seed=seed)

    layout: KVCacheLayout = case["layout"]
    cache_pool: KVCachePool = case["cache_pool"]
    layer_id: int = case["layer_id"]
    block_table: list[int] = case["block_table"]
    block_table_tensor: torch.Tensor = case["block_table_tensor"]
    q: torch.Tensor = case["q"]

    def run_reference() -> torch.Tensor:
        return paged_attention_decode_reference(
            q=q,
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_table=block_table,
            seq_len=seq_len,
        )

    def run_cuda() -> torch.Tensor:
        return paged_attention_cuda.paged_attention_decode(
            q,
            cache_pool.key_cache,
            cache_pool.value_cache,
            block_table_tensor,
            layer_id,
            seq_len,
        )

    reference_output = run_reference()
    cuda_output = run_cuda()

    diff = cuda_output.float() - reference_output.float()
    max_abs_diff = diff.abs().max().item()
    passed = (
        torch.isfinite(reference_output).all().item()
        and torch.isfinite(cuda_output).all().item()
        and torch.isfinite(diff).all().item()
        and max_abs_diff < 1e-2
    )

    reference_ms = benchmark_cuda_events(
        fn=run_reference,
        warmup_iters=WARMUP_ITERS,
        bench_iters=BENCH_ITERS,
    )

    cuda_ms = benchmark_cuda_events(
        fn=run_cuda,
        warmup_iters=WARMUP_ITERS,
        bench_iters=BENCH_ITERS,
    )

    speedup_vs_reference = reference_ms / cuda_ms if cuda_ms > 0 else float("inf")

    return {
        "kernel_name": KERNEL_NAME,
        "seq_len": seq_len,
        "seed": seed,
        "passed": bool(passed),
        "max_abs_diff": max_abs_diff,
        "reference_ms": reference_ms,
        "cuda_ms": cuda_ms,
        "speedup_vs_reference": speedup_vs_reference,
        "block_table": block_table,
        "num_blocks": len(block_table),
        "layout": layout.snapshot(),
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
    }


def write_json_report(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bench_name": BENCH_NAME,
        "kernel_name": KERNEL_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "loaded_extension": str(paged_attention_cuda.__file__),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
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

    lines.append(f"# {BENCH_NAME}")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Benchmark the CUDA paged attention decode kernel against the PyTorch "
        "reference implementation across increasing sequence lengths."
    )
    lines.append("")
    lines.append("This benchmark is intended for iteration guidance, not final production performance claims.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Loaded extension: `{paged_attention_cuda.__file__}`")
    lines.append(f"- PyTorch: `{torch.__version__}`")
    lines.append(f"- CUDA: `{torch.version.cuda}`")
    lines.append(f"- Device: `{torch.cuda.get_device_name(0)}`")
    lines.append("")
    lines.append("## Benchmark Config")
    lines.append("")
    lines.append(f"- Warmup iterations: `{WARMUP_ITERS}`")
    lines.append(f"- Benchmark iterations: `{BENCH_ITERS}`")
    lines.append(f"- Block size tokens: `{BLOCK_SIZE_TOKENS}`")
    lines.append(f"- Total blocks: `{TOTAL_BLOCKS}`")
    lines.append(f"- Num KV heads: `{NUM_KV_HEADS}`")
    lines.append(f"- Head dim: `{HEAD_DIM}`")
    lines.append(f"- Dtype: `{DTYPE}`")
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
    lines.append("## Results")
    lines.append("")
    lines.append("| seq_len | blocks | max_abs_diff | reference ms | cuda ms | speedup | passed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")

    for result in results:
        lines.append(
            f"| {result['seq_len']} "
            f"| {result['num_blocks']} "
            f"| {result['max_abs_diff']:.8f} "
            f"| {result['reference_ms']:.6f} "
            f"| {result['cuda_ms']:.6f} "
            f"| {result['speedup_vs_reference']:.2f}x "
            f"| {result['passed']} |"
        )

    lines.append("")
    lines.append("## Correctness")
    lines.append("")
    if all_passed:
        lines.append("All benchmark cases passed correctness checks before timing.")
    else:
        lines.append("One or more benchmark cases failed correctness checks.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The PyTorch reference is intentionally simple and includes Python-level looping over the paged KV cache. "
        "The CUDA kernel should increasingly benefit as sequence length grows, but this v2 kernel still recomputes "
        "QK scores multiple times and is not expected to represent final performance."
    )
    lines.append("")
    lines.append("## Next Kernel Question")
    lines.append("")
    lines.append(
        "If v2 is not significantly faster, the likely bottleneck is repeated QK score recomputation. "
        "The next iteration should consider storing scores or using an online softmax structure."
    )
    lines.append("")

    output_path.write_text("\n".join(lines))


def print_terminal_summary(results: list[dict[str, Any]]) -> None:
    print(BENCH_NAME)
    print("---")
    print("loaded extension:", paged_attention_cuda.__file__)
    print("seq_len | blocks | max_abs_diff | reference_ms | cuda_ms | speedup | passed")
    print("--- | --- | --- | --- | --- | --- | ---")

    for result in results:
        print(
            f"{result['seq_len']} | "
            f"{result['num_blocks']} | "
            f"{result['max_abs_diff']:.8f} | "
            f"{result['reference_ms']:.6f} | "
            f"{result['cuda_ms']:.6f} | "
            f"{result['speedup_vs_reference']:.2f}x | "
            f"{result['passed']}"
        )

    print()
    print("all_passed:", all(result["passed"] for result in results))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    results = []

    for index, seq_len in enumerate(SEQ_LENS):
        result = run_one_case(
            seq_len=seq_len,
            seed=index,
        )
        results.append(result)

    print_terminal_summary(results)

    results_json_path = Path("results") / f"{BENCH_NAME}.json"
    markdown_path = (
        PROJECT_ROOT
        / "docs"
        / "kernel_iterations"
        / f"{BENCH_NAME}.md"
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