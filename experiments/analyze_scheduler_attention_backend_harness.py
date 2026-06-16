from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def percentile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results/scheduler_attention_backend_harness.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/runtime/scheduler_attention_backend_harness_report.md",
    )
    args = parser.parse_args()

    csv_path = REPO_ROOT / args.csv
    out_path = REPO_ROOT / args.out

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "decode_step",
        "active_batch_size",
        "backend",
        "backend_ms",
        "kv_used_blocks",
        "kv_free_blocks",
        "kv_utilization",
        "total_tokens_emitted",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Harness CSV is empty")

    backend = str(df["backend"].iloc[0])
    decode_steps = int(df["decode_step"].max()) + 1
    total_tokens_emitted = int(df["total_tokens_emitted"].max())

    backend_ms = df["backend_ms"]
    active_batch_size = df["active_batch_size"]
    kv_utilization = df["kv_utilization"]

    summary = {
        "backend": backend,
        "decode_steps": decode_steps,
        "total_tokens_emitted": total_tokens_emitted,
        "backend_med_ms": float(backend_ms.median()),
        "backend_min_ms": float(backend_ms.min()),
        "backend_max_ms": float(backend_ms.max()),
        "backend_p95_ms": percentile(backend_ms, 0.95),
        "active_batch_avg": float(active_batch_size.mean()),
        "active_batch_max": int(active_batch_size.max()),
        "kv_utilization_initial": float(kv_utilization.iloc[0]),
        "kv_utilization_final": float(kv_utilization.iloc[-1]),
        "kv_utilization_max": float(kv_utilization.max()),
    }

    tail_table = df.tail(10).to_string(index=False)

    lines = [
        "# Scheduler Attention Backend Harness Report",
        "",
        "## Source",
        "",
        "```text",
        str(csv_path.relative_to(REPO_ROOT)),
        "```",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Backend | `{summary['backend']}` |",
        f"| Decode steps | {summary['decode_steps']} |",
        f"| Total tokens emitted | {summary['total_tokens_emitted']} |",
        f"| Backend median latency | {summary['backend_med_ms']:.6f} ms |",
        f"| Backend min latency | {summary['backend_min_ms']:.6f} ms |",
        f"| Backend p95 latency | {summary['backend_p95_ms']:.6f} ms |",
        f"| Backend max latency | {summary['backend_max_ms']:.6f} ms |",
        f"| Average active batch size | {summary['active_batch_avg']:.2f} |",
        f"| Max active batch size | {summary['active_batch_max']} |",
        f"| Initial KV utilization | {summary['kv_utilization_initial']:.4f} |",
        f"| Max KV utilization | {summary['kv_utilization_max']:.4f} |",
        f"| Final KV utilization | {summary['kv_utilization_final']:.4f} |",
        "",
        "## Interpretation",
        "",
        "This report summarizes the scheduler-owned synthetic decode harness.",
        "",
        "The harness validates this runtime path:",
        "",
        "```text",
        "RequestState",
        "  -> KVBlockManager",
        "  -> DecodeBatch",
        "  -> KVCachePool",
        "  -> AttentionBackend",
        "  -> CUDA paged attention kernel",
        "```",
        "",
        "This is not full model execution. Query tensors, generated token IDs, and generated K/V entries are synthetic.",
        "The purpose is to validate scheduler/KV/backend plumbing and measure the attention backend inside a repeated decode loop.",
        "",
        "## Per-Step Tail",
        "",
        "Last 10 decode steps:",
        "",
        "```text",
        tail_table,
        "```",
        "",
    ]

    markdown = "\n".join(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)

    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()