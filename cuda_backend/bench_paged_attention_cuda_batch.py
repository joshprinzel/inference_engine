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


KERNEL_NAME = "paged_attention_cuda_v4_batched"
BENCH_NAME = f"{KERNEL_NAME}_batch_sweep_bench"

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
SEQ_LENS = [128, 256, 512]

NUM_LAYERS = 2
TOTAL_BLOCKS = 32768
BLOCK_SIZE_TOKENS = 8
NUM_KV_HEADS = 16
HEAD_DIM = 128
DTYPE = "float16"
DEVICE = "cuda"

DEFAULT_WARMUP_ITERS = 25
TRIALS = 5


def bench_iters_for_case(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    work = batch_size * seq_len * num_kv_heads * head_dim

    if work <= 4 * 128 * 16 * 128:
        return 300

    if work <= 16 * 256 * 16 * 128:
        return 100

    return 50


def summarize_measurements(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n % 2 == 1:
        median = sorted_values[n // 2]
    else:
        median = 0.5 * (sorted_values[n // 2 - 1] + sorted_values[n // 2])

    return {
        "min": sorted_values[0],
        "median": median,
        "max": sorted_values[-1],
    }


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
    batch_size: int,
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
    seq_lens = [seq_len] * batch_size
    block_tables: list[list[int]] = []

    for request_index in range(batch_size):
        request_id = f"batch-bench-req-{seed}-{request_index}"

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
        "batch_size": batch_size,
        "seq_len": seq_len,
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


def benchmark_trials(
    fn,
    warmup_iters: int,
    bench_iters: int,
    trials: int,
) -> dict[str, Any]:
    trial_values = []

    for _ in range(trials):
        value = benchmark_cuda_events(
            fn=fn,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        trial_values.append(value)

    summary = summarize_measurements(trial_values)

    return {
        "trials": trial_values,
        "min": summary["min"],
        "median": summary["median"],
        "max": summary["max"],
    }


def run_one_case(
    batch_size: int,
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    case = make_case(
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
    )

    layout: KVCacheLayout = case["layout"]
    cache_pool: KVCachePool = case["cache_pool"]
    layer_id: int = case["layer_id"]
    seq_lens: list[int] = case["seq_lens"]
    block_tables: list[list[int]] = case["block_tables"]
    block_tables_tensor: torch.Tensor = case["block_tables_tensor"]
    seq_lens_tensor: torch.Tensor = case["seq_lens_tensor"]
    q: torch.Tensor = case["q"]

    warmup_iters = DEFAULT_WARMUP_ITERS
    bench_iters = bench_iters_for_case(
        batch_size=batch_size,
        seq_len=seq_len,
        num_kv_heads=layout.num_kv_heads,
        head_dim=layout.head_dim,
    )

    def run_reference() -> torch.Tensor:
        return reference_batch_decode(
            q=q,
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )

    def run_cuda() -> torch.Tensor:
        return paged_attention_cuda.paged_attention_decode_batch(
            q,
            cache_pool.key_cache,
            cache_pool.value_cache,
            block_tables_tensor,
            seq_lens_tensor,
            layer_id,
        )

    reference_output = run_reference()
    cuda_output = run_cuda()

    diff = cuda_output.float() - reference_output.float()
    max_abs_diff = diff.abs().max().item()

    reference_finite = torch.isfinite(reference_output).all().item()
    cuda_finite = torch.isfinite(cuda_output).all().item()
    diff_finite = torch.isfinite(diff).all().item()

    passed = (
        reference_finite
        and cuda_finite
        and diff_finite
        and max_abs_diff < 1e-2
    )

    reference_timing = benchmark_trials(
        fn=run_reference,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        trials=TRIALS,
    )

    cuda_timing = benchmark_trials(
        fn=run_cuda,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        trials=TRIALS,
    )

    reference_ms = reference_timing["median"]
    cuda_ms = cuda_timing["median"]

    speedup_vs_reference = reference_ms / cuda_ms if cuda_ms > 0 else float("inf")

    num_ctas = batch_size * layout.num_kv_heads
    attended_tokens = batch_size * seq_len
    attention_elements = batch_size * seq_len * layout.num_kv_heads * layout.head_dim

    requests_per_ms = batch_size / cuda_ms if cuda_ms > 0 else float("inf")
    attended_tokens_per_ms = attended_tokens / cuda_ms if cuda_ms > 0 else float("inf")
    attention_elements_per_ms = attention_elements / cuda_ms if cuda_ms > 0 else float("inf")

    return {
        "kernel_name": KERNEL_NAME,
        "seed": seed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_kv_heads": layout.num_kv_heads,
        "head_dim": layout.head_dim,
        "num_ctas": num_ctas,
        "passed": bool(passed),
        "max_abs_diff": max_abs_diff,
        "reference_finite": bool(reference_finite),
        "cuda_finite": bool(cuda_finite),
        "diff_finite": bool(diff_finite),
        "reference_ms": reference_ms,
        "cuda_ms": cuda_ms,
        "reference_ms_min": reference_timing["min"],
        "reference_ms_median": reference_timing["median"],
        "reference_ms_max": reference_timing["max"],
        "reference_ms_trials": reference_timing["trials"],
        "cuda_ms_min": cuda_timing["min"],
        "cuda_ms_median": cuda_timing["median"],
        "cuda_ms_max": cuda_timing["max"],
        "cuda_ms_trials": cuda_timing["trials"],
        "speedup_vs_reference": speedup_vs_reference,
        "requests_per_ms": requests_per_ms,
        "attended_tokens_per_ms": attended_tokens_per_ms,
        "attention_elements_per_ms": attention_elements_per_ms,
        "block_tables_shape": tuple(block_tables_tensor.shape),
        "num_blocks_per_request": len(block_tables[0]),
        "layout": layout.snapshot(),
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "trials": TRIALS,
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
        "batch_sizes": BATCH_SIZES,
        "seq_lens": SEQ_LENS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "block_size_tokens": BLOCK_SIZE_TOKENS,
        "total_blocks": TOTAL_BLOCKS,
        "dtype": DTYPE,
        "default_warmup_iters": DEFAULT_WARMUP_ITERS,
        "trials": TRIALS,
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

    lines = []

    lines.append(f"# {BENCH_NAME}")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Benchmark the batched CUDA paged attention decode kernel across increasing "
        "batch sizes and sequence lengths."
    )
    lines.append("")
    lines.append(
        "This benchmark tests whether exposing more sequence/head CTAs improves GPU occupancy "
        "and serving-shaped throughput."
    )
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
    lines.append(f"- Batch sizes: `{BATCH_SIZES}`")
    lines.append(f"- Sequence lengths: `{SEQ_LENS}`")
    lines.append(f"- Num KV heads: `{NUM_KV_HEADS}`")
    lines.append(f"- Head dim: `{HEAD_DIM}`")
    lines.append(f"- Block size tokens: `{BLOCK_SIZE_TOKENS}`")
    lines.append(f"- Total blocks: `{TOTAL_BLOCKS}`")
    lines.append(f"- Dtype: `{DTYPE}`")
    lines.append(f"- Default warmup iterations: `{DEFAULT_WARMUP_ITERS}`")
    lines.append(f"- Trials per case: `{TRIALS}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| batch | seq_len | CTAs | blocks/req | max_abs_diff | ref median ms | cuda median ms | cuda min ms | cuda max ms | speedup | req/ms | attended tok/ms | elems/ms | iters | passed |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for result in results:
        lines.append(
            f"| {result['batch_size']} "
            f"| {result['seq_len']} "
            f"| {result['num_ctas']} "
            f"| {result['num_blocks_per_request']} "
            f"| {result['max_abs_diff']:.8f} "
            f"| {result['reference_ms_median']:.6f} "
            f"| {result['cuda_ms_median']:.6f} "
            f"| {result['cuda_ms_min']:.6f} "
            f"| {result['cuda_ms_max']:.6f} "
            f"| {result['speedup_vs_reference']:.2f}x "
            f"| {result['requests_per_ms']:.2f} "
            f"| {result['attended_tokens_per_ms']:.2f} "
            f"| {result['attention_elements_per_ms']:.2f} "
            f"| {result['bench_iters']} "
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
    lines.append("## Timing Method")
    lines.append("")
    lines.append(
        "Each row reports the median of multiple CUDA-event timing trials. "
        "The minimum and maximum CUDA timings are included to expose benchmark variance."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The v4 kernel launches one CUDA block per sequence/head pair. "
        "Therefore, the number of CTAs scales as `batch_size * num_heads`. "
        "If the single-request kernel was under-occupying the GPU, throughput should improve as batch size increases."
    )
    lines.append("")
    lines.append("## Next Kernel Question")
    lines.append("")
    lines.append(
        "If throughput saturates at moderate batch size, the next bottlenecks are likely scalar memory access, "
        "serial softmax denominator computation, or inefficient per-head CTA work decomposition."
    )
    lines.append("")

    output_path.write_text("\n".join(lines))


def print_terminal_summary(results: list[dict[str, Any]]) -> None:
    print(BENCH_NAME)
    print("---")
    print("loaded extension:", paged_attention_cuda.__file__)
    print(
        "batch | seq_len | CTAs | blocks/req | max_abs_diff | ref_med_ms | cuda_med_ms | cuda_min_ms | cuda_max_ms | speedup | req/ms | attended_tok/ms | elems/ms | iters | passed"
    )
    print(
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"
    )

    for result in results:
        print(
            f"{result['batch_size']} | "
            f"{result['seq_len']} | "
            f"{result['num_ctas']} | "
            f"{result['num_blocks_per_request']} | "
            f"{result['max_abs_diff']:.8f} | "
            f"{result['reference_ms_median']:.6f} | "
            f"{result['cuda_ms_median']:.6f} | "
            f"{result['cuda_ms_min']:.6f} | "
            f"{result['cuda_ms_max']:.6f} | "
            f"{result['speedup_vs_reference']:.2f}x | "
            f"{result['requests_per_ms']:.2f} | "
            f"{result['attended_tokens_per_ms']:.2f} | "
            f"{result['attention_elements_per_ms']:.2f} | "
            f"{result['bench_iters']} | "
            f"{result['passed']}"
        )

    print()
    print("all_passed:", all(result["passed"] for result in results))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    results = []
    case_index = 0

    for seq_len in SEQ_LENS:
        for batch_size in BATCH_SIZES:
            result = run_one_case(
                batch_size=batch_size,
                seq_len=seq_len,
                seed=case_index,
            )
            results.append(result)
            case_index += 1

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