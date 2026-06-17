import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.paged_attention_reference import paged_attention_decode_reference

import paged_attention_cuda


KERNEL_NAME = "paged_attention_cuda_v9a_gqa_mqa"
BENCH_NAME = f"{KERNEL_NAME}_bench"

ATTENTION_CONFIGS = [
    {
        "name": "mha",
        "num_query_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 128,
    },
    {
        "name": "gqa",
        "num_query_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
    {
        "name": "mqa",
        "num_query_heads": 16,
        "num_kv_heads": 1,
        "head_dim": 128,
    },
]

NUM_LAYERS = 2
TOTAL_BLOCKS = 32768
BLOCK_SIZE_TOKENS = 8
DTYPE = "float16"
DEVICE = "cuda"
MAX_ALLOWED_DIFF = 1e-2


@dataclass(frozen=True)
class BenchProfile:
    name: str
    batch_sizes: list[int]
    seq_lens: list[int]
    warmup_iters: int
    trials: int
    time_reference: bool
    bench_iter_scale: float = 1.0


BENCH_PROFILES = {
    # Fast iteration profile. Keeps attention-mode coverage and representative
    # latency/throughput points while avoiding redundant middle grid points.
    "quick": BenchProfile(
        name="quick",
        batch_sizes=[1, 8, 32],
        seq_lens=[128, 512],
        warmup_iters=10,
        trials=3,
        time_reference=False,
        bench_iter_scale=1.0,
    ),
    # Original exhaustive grid with full reference timing.
    "full": BenchProfile(
        name="full",
        batch_sizes=[1, 4, 8, 16, 32],
        seq_lens=[128, 256, 512],
        warmup_iters=25,
        trials=5,
        time_reference=True,
        bench_iter_scale=1.0,
    ),
    # Useful when you want correctness plus a very quick smoke-test timing pass.
    "smoke": BenchProfile(
        name="smoke",
        batch_sizes=[1, 32],
        seq_lens=[128, 512],
        warmup_iters=5,
        trials=2,
        time_reference=False,
        bench_iter_scale=0.5,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched CUDA paged attention decode across MHA/GQA/MQA."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(BENCH_PROFILES),
        default="quick",
        help="Benchmark profile. Default: quick.",
    )
    parser.add_argument(
        "--time-reference",
        action="store_true",
        help="Also benchmark the Python reference implementation.",
    )
    parser.add_argument(
        "--skip-reference-timing",
        action="store_true",
        help="Do not benchmark the Python reference implementation, even for full profile.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=None,
        help="Override warmup iterations per timing trial.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override timing trials per case.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix for report files. Default: profile name.",
    )
    return parser.parse_args()


def resolve_profile(args: argparse.Namespace) -> BenchProfile:
    base = BENCH_PROFILES[args.profile]

    if args.time_reference and args.skip_reference_timing:
        raise ValueError("Use either --time-reference or --skip-reference-timing, not both")

    time_reference = base.time_reference
    if args.time_reference:
        time_reference = True
    if args.skip_reference_timing:
        time_reference = False

    return BenchProfile(
        name=base.name,
        batch_sizes=base.batch_sizes,
        seq_lens=base.seq_lens,
        warmup_iters=args.warmup_iters if args.warmup_iters is not None else base.warmup_iters,
        trials=args.trials if args.trials is not None else base.trials,
        time_reference=time_reference,
        bench_iter_scale=base.bench_iter_scale,
    )


def bench_iters_for_case(
    batch_size: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float = 1.0,
) -> int:
    # This is a rough iteration budget. Query-head work dominates output size;
    # KV-head count affects cache footprint and K/V indexing.
    work = batch_size * seq_len * num_query_heads * head_dim

    if work <= 4 * 128 * 16 * 128:
        base_iters = 300
    elif work <= 16 * 256 * 16 * 128:
        base_iters = 100
    else:
        base_iters = 50

    return max(1, int(round(base_iters * scale)))


def summarize_measurements(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": math.nan,
            "median": math.nan,
            "max": math.nan,
        }

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


def empty_timing() -> dict[str, Any]:
    return {
        "trials": [],
        "min": math.nan,
        "median": math.nan,
        "max": math.nan,
    }


def format_float(value: float, precision: int = 6) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{precision}f}"


def format_speedup(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


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
    attention_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    if num_query_heads < num_kv_heads:
        raise ValueError("num_query_heads must be >= num_kv_heads")

    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    layout = KVCacheLayout(
        num_layers=NUM_LAYERS,
        total_blocks=TOTAL_BLOCKS,
        block_size_tokens=BLOCK_SIZE_TOKENS,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
        request_id = (
            f"{attention_name}-bench-req-"
            f"seed{seed}-b{request_index}-s{seq_len}"
        )

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
        num_query_heads,
        head_dim,
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
        "attention_name": attention_name,
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
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
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
    fn: Callable[[], torch.Tensor],
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
    fn: Callable[[], torch.Tensor],
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
    attention_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    profile: BenchProfile,
) -> dict[str, Any]:
    case = make_case(
        attention_name=attention_name,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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

    warmup_iters = profile.warmup_iters
    bench_iters = bench_iters_for_case(
        batch_size=batch_size,
        seq_len=seq_len,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=profile.bench_iter_scale,
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

    # Always run the reference once for correctness, even in quick/smoke modes.
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
        and max_abs_diff < MAX_ALLOWED_DIFF
    )

    if profile.time_reference:
        reference_timing = benchmark_trials(
            fn=run_reference,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            trials=profile.trials,
        )
    else:
        reference_timing = empty_timing()

    cuda_timing = benchmark_trials(
        fn=run_cuda,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        trials=profile.trials,
    )

    reference_ms = reference_timing["median"]
    cuda_ms = cuda_timing["median"]

    if profile.time_reference and cuda_ms > 0:
        speedup_vs_reference = reference_ms / cuda_ms
    else:
        speedup_vs_reference = math.nan

    num_ctas = batch_size * num_query_heads
    attended_tokens = batch_size * seq_len
    query_attention_elements = batch_size * seq_len * num_query_heads * head_dim
    kv_cache_elements = batch_size * seq_len * num_kv_heads * head_dim

    requests_per_ms = batch_size / cuda_ms if cuda_ms > 0 else float("inf")
    attended_tokens_per_ms = attended_tokens / cuda_ms if cuda_ms > 0 else float("inf")
    query_attention_elements_per_ms = (
        query_attention_elements / cuda_ms if cuda_ms > 0 else float("inf")
    )
    kv_cache_elements_per_ms = (
        kv_cache_elements / cuda_ms if cuda_ms > 0 else float("inf")
    )

    return {
        "kernel_name": KERNEL_NAME,
        "profile": profile.name,
        "time_reference": profile.time_reference,
        "attention_name": attention_name,
        "seed": seed,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "query_heads_per_kv_head": num_query_heads // num_kv_heads,
        "head_dim": head_dim,
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
        "query_attention_elements_per_ms": query_attention_elements_per_ms,
        "kv_cache_elements_per_ms": kv_cache_elements_per_ms,
        "block_tables_shape": tuple(block_tables_tensor.shape),
        "num_blocks_per_request": len(block_tables[0]),
        "layout": layout.snapshot(),
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
        "trials": profile.trials,
    }


def write_json_report(
    results: list[dict[str, Any]],
    output_path: Path,
    profile: BenchProfile,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bench_name": BENCH_NAME,
        "kernel_name": KERNEL_NAME,
        "profile": profile.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "loaded_extension": str(paged_attention_cuda.__file__),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "attention_configs": ATTENTION_CONFIGS,
        "batch_sizes": profile.batch_sizes,
        "seq_lens": profile.seq_lens,
        "block_size_tokens": BLOCK_SIZE_TOKENS,
        "total_blocks": TOTAL_BLOCKS,
        "dtype": DTYPE,
        "warmup_iters": profile.warmup_iters,
        "trials": profile.trials,
        "time_reference": profile.time_reference,
        "bench_iter_scale": profile.bench_iter_scale,
        "num_cases": len(results),
        "all_passed": all(result["passed"] for result in results),
        "results": results,
    }

    output_path.write_text(json.dumps(payload, indent=2, allow_nan=True))


def write_markdown_report(
    results: list[dict[str, Any]],
    output_path: Path,
    profile: BenchProfile,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_passed = all(result["passed"] for result in results)

    lines = []

    lines.append(f"# {BENCH_NAME}")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Benchmark the batched CUDA paged attention decode kernel across MHA, GQA, and MQA configurations."
    )
    lines.append("")
    lines.append(
        "This benchmark tests the performance effect of reducing KV heads while keeping query heads fixed."
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
    lines.append(f"- Profile: `{profile.name}`")
    lines.append(f"- Attention configs: `{ATTENTION_CONFIGS}`")
    lines.append(f"- Batch sizes: `{profile.batch_sizes}`")
    lines.append(f"- Sequence lengths: `{profile.seq_lens}`")
    lines.append(f"- Number of cases: `{len(results)}`")
    lines.append(f"- Block size tokens: `{BLOCK_SIZE_TOKENS}`")
    lines.append(f"- Total blocks: `{TOTAL_BLOCKS}`")
    lines.append(f"- Dtype: `{DTYPE}`")
    lines.append(f"- Warmup iterations: `{profile.warmup_iters}`")
    lines.append(f"- Trials per case: `{profile.trials}`")
    lines.append(f"- Reference timing enabled: `{profile.time_reference}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| mode | batch | seq_len | q_heads | kv_heads | q/kv | CTAs | blocks/req | max_abs_diff | ref med ms | cuda med ms | cuda min ms | cuda max ms | speedup | req/ms | attended tok/ms | q-elems/ms | kv-elems/ms | iters | passed |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for result in results:
        lines.append(
            f"| {result['attention_name']} "
            f"| {result['batch_size']} "
            f"| {result['seq_len']} "
            f"| {result['num_query_heads']} "
            f"| {result['num_kv_heads']} "
            f"| {result['query_heads_per_kv_head']} "
            f"| {result['num_ctas']} "
            f"| {result['num_blocks_per_request']} "
            f"| {result['max_abs_diff']:.8f} "
            f"| {format_float(result['reference_ms_median'])} "
            f"| {format_float(result['cuda_ms_median'])} "
            f"| {format_float(result['cuda_ms_min'])} "
            f"| {format_float(result['cuda_ms_max'])} "
            f"| {format_speedup(result['speedup_vs_reference'])} "
            f"| {format_float(result['requests_per_ms'], 2)} "
            f"| {format_float(result['attended_tokens_per_ms'], 2)} "
            f"| {format_float(result['query_attention_elements_per_ms'], 2)} "
            f"| {format_float(result['kv_cache_elements_per_ms'], 2)} "
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
    if not profile.time_reference:
        lines.append("")
        lines.append(
            "Reference timing was skipped for this profile. The reference implementation was still run once per case for correctness."
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "MHA uses one KV head per query head. GQA shares each KV head across multiple query heads. "
        "MQA shares one KV head across all query heads. Reducing KV heads reduces KV cache storage and changes the memory-access pattern, "
        "while the number of query heads still determines the number of sequence/head CTAs launched."
    )
    lines.append("")
    lines.append("## Next Kernel Question")
    lines.append("")
    lines.append(
        "If GQA/MQA performance does not improve despite fewer KV heads, the bottleneck is likely not raw KV-cache footprint yet. "
        "The next optimization pass should use profiling to inspect scalar V loads, serial softmax denominator computation, and CTA-level occupancy."
    )
    lines.append("")

    output_path.write_text("\n".join(lines))


def print_terminal_summary(results: list[dict[str, Any]], profile: BenchProfile) -> None:
    print(BENCH_NAME)
    print("---")
    print("profile:", profile.name)
    print("loaded extension:", paged_attention_cuda.__file__)
    print(
        "mode | batch | seq_len | q_heads | kv_heads | q/kv | CTAs | blocks/req | max_abs_diff | ref_med_ms | cuda_med_ms | cuda_min_ms | cuda_max_ms | speedup | req/ms | attended_tok/ms | q_elems/ms | kv_elems/ms | iters | passed"
    )
    print(
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"
    )

    for result in results:
        print(
            f"{result['attention_name']} | "
            f"{result['batch_size']} | "
            f"{result['seq_len']} | "
            f"{result['num_query_heads']} | "
            f"{result['num_kv_heads']} | "
            f"{result['query_heads_per_kv_head']} | "
            f"{result['num_ctas']} | "
            f"{result['num_blocks_per_request']} | "
            f"{result['max_abs_diff']:.8f} | "
            f"{format_float(result['reference_ms_median'])} | "
            f"{format_float(result['cuda_ms_median'])} | "
            f"{format_float(result['cuda_ms_min'])} | "
            f"{format_float(result['cuda_ms_max'])} | "
            f"{format_speedup(result['speedup_vs_reference'])} | "
            f"{format_float(result['requests_per_ms'], 2)} | "
            f"{format_float(result['attended_tokens_per_ms'], 2)} | "
            f"{format_float(result['query_attention_elements_per_ms'], 2)} | "
            f"{format_float(result['kv_cache_elements_per_ms'], 2)} | "
            f"{result['bench_iters']} | "
            f"{result['passed']}"
        )

    print()
    print("all_passed:", all(result["passed"] for result in results))


def report_paths(profile: BenchProfile, output_suffix: str | None) -> tuple[Path, Path]:
    suffix = output_suffix if output_suffix is not None else profile.name
    filename = f"{BENCH_NAME}_{suffix}"

    results_json_path = Path("results") / f"{filename}.json"
    markdown_path = (
        PROJECT_ROOT
        / "docs"
        / "kernel_iterations"
        / f"{filename}.md"
    )

    return results_json_path, markdown_path


def main() -> None:
    args = parse_args()
    profile = resolve_profile(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    results = []
    case_index = 0

    for attention_config in ATTENTION_CONFIGS:
        for seq_len in profile.seq_lens:
            for batch_size in profile.batch_sizes:
                result = run_one_case(
                    attention_name=attention_config["name"],
                    num_query_heads=attention_config["num_query_heads"],
                    num_kv_heads=attention_config["num_kv_heads"],
                    head_dim=attention_config["head_dim"],
                    batch_size=batch_size,
                    seq_len=seq_len,
                    seed=case_index,
                    profile=profile,
                )

                results.append(result)
                case_index += 1

    print_terminal_summary(results, profile=profile)

    results_json_path, markdown_path = report_paths(
        profile=profile,
        output_suffix=args.output_suffix,
    )

    write_json_report(
        results=results,
        output_path=results_json_path,
        profile=profile,
    )

    write_markdown_report(
        results=results,
        output_path=markdown_path,
        profile=profile,
    )

    print()
    print(f"wrote JSON: {results_json_path.resolve()}")
    print(f"wrote markdown: {markdown_path.resolve()}")

    assert all(result["passed"] for result in results)


if __name__ == "__main__":
    main()
