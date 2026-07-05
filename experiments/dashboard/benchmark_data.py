from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_BENCHMARK_DIR = Path("results/benchmarks")

def list_benchmark_csvs(benchmark_dir: Path = DEFAULT_BENCHMARK_DIR) -> list[Path]:
    if not benchmark_dir.exists():
        return []
    
    return sorted(
        benchmark_dir.rglob("runtime_benchmark_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

def infer_benchmark_path_metadata(
    *,
    csv_path: Path,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
) -> dict[str, str]:
    """
    Infer scenario/policy/matrix metadata from a benchmark CSV path.

    Policy matrix outputs are expected to look like:

      results/benchmarks/policy_matrix_<timestamp>/<scenario>/<policy>/runtime_benchmark_*.csv

    Older flat benchmark CSVs are still supported and get fallback metadata.
    """

    try:
        relative_parts = csv_path.relative_to(benchmark_dir).parts
    except ValueError:
        relative_parts = csv_path.parts

    metadata = {
        "benchmark_file": csv_path.name,
        "benchmark_path": str(csv_path),
        "matrix_run": "unknown",
        "scenario_name": "manual",
        "policy_dir": "manual",
    }

    if len(relative_parts) >= 4 and relative_parts[-1].startswith("runtime_benchmark_"):
        metadata["matrix_run"] = relative_parts[-4]
        metadata["scenario_name"] = relative_parts[-3]
        metadata["policy_dir"] = relative_parts[-2]

    return metadata 


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_columns = [
        "repeat_index",
        "num_requests",
        "max_slots",
        "max_new_tokens",
        "max_decode_batch_size",
        "block_size_tokens",
        "total_kv_blocks",
        "total_wall_seconds",
        "tokens_generated",
        "tokens_per_second",
        "avg_queue_wait_ms",
        "avg_ttft_ms",
        "avg_decode_latency_ms",
        "avg_latency_ms",
        "decode_iterations",
        "decode_batches_built",
        "admitted_count",
        "decode_stalls",
        "kv_allocation_failures",
        "kv_oom_evictions",
        "late_admissions",
        "early_finishes",
        "backend_ms_median",
        "backend_ms_p95",
        "backend_ms_min",
        "backend_ms_max",
        "backend_ms_mean",
        "kv_peak_used_blocks",
        "kv_final_used_blocks",
        "kv_final_free_blocks",
        "kv_final_utilization",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    
    if "correctness_passed" in df.columns:
        df["correctness_passed"] = (
            df["correctness_passed"].astype(str).str.lower().eq("true")
        )
    
    if "all_finished" in df.columns:
        df["all_finished"] = df["all_finished"].astype(str).str.lower().eq("true")

    return df

def load_benchmark_csvs(
        csv_paths: list[Path],
        *,
        benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
) -> pd.DataFrame:
    """
    Load and concatenate benchmark CSVs with metadata inferred from paths.
    """

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = load_benchmark_csv(
            csv_path
        )
        metadata = infer_benchmark_path_metadata(
            csv_path=csv_path,
            benchmark_dir=benchmark_dir
        )
        for key, value in metadata.items():
            df[key] = value
        
        frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def filter_measured_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "run_kind" not in df.columns:
        return df
    
    return df[df["run_kind"] == "measured"].copy()



def aggregate_medians(
        df: pd.DataFrame,
        group_columns: list[str]
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    
    metrics_column = [
        "tokens_per_second",
        "avg_queue_wait_ms",
        "avg_ttft_ms",
        "avg_decode_latency_ms",
        "avg_latency_ms",
        "backend_ms_median",
        "backend_ms_p95",
        "backend_ms_mean",
        "kv_peak_used_blocks",
        "decode_iterations",
        "decode_batches_built",
    ]

    available_metrics = [column for column in metrics_column if column in df.columns]

    aggregated = (
        df.groupby(group_columns, as_index=False)[available_metrics]
        .median()
        .sort_values(group_columns)
    )

    if "correctness_passed" in df.columns:
        correctness = (
            df.groupby(group_columns, as_index=False)["correctness_passed"]
            .all()
            .rename(columns={"correctness_passed": "correctness_all_passed"})

        )
        aggregated = aggregated.merge(correctness, on=group_columns, how="left")
    if "all_finished" in df.columns:
        all_finished = (
            df.groupby(group_columns, as_index=False)["all_finished"]
            .all()
            .rename(columns={"all_finished": "all_finished_all_passed"})
        )
        aggregated = aggregated.merge(all_finished, on=group_columns, how="left")
    return aggregated