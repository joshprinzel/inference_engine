from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_BENCHMARK_DIR = Path("results/benchmarks")

def list_benchmark_csvs(benchmark_dir: Path = DEFAULT_BENCHMARK_DIR) -> list[Path]:
    if not benchmark_dir.exists():
        return []
    
    return sorted(
        benchmark_dir.glob("runtime_benchmark_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_columns = [
        "repeat_index",
        "num_requests",
        "max_slots",
        "max_new_tokens",
        "block_size_tokens",
        "total_kv_blocks",
        "total_wall_seconds",
        "tokens_generated",
        "tokens_per_second",
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
        "backend_ms_median",
        "backend_ms_p95",
        "backend_ms_mean",
        "kv_peak_used_blocks",
        "decode_iterations",
        "decode_batches_built"
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
    return aggregated