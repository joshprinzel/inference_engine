from __future__ import annotations

import csv
from pathlib import Path

from experiments.analysis.plot_runtime_benchmarks import main


def test_plot_runtime_benchmarks_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "benchmark.csv"
    output_dir = tmp_path / "plots"

    fieldnames = [
        "run_kind",
        "repeat_index",
        "backend",
        "num_requests",
        "max_slots",
        "max_new_tokens",
        "block_size_tokens",
        "total_kv_blocks",
        "dtype",
        "device",
        "prompt_set",
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
        "all_finished",
        "generated_text_by_request",
        "expected_text_by_request",
        "correctness_passed",
    ]

    rows = [
        {
            "run_kind": "measured",
            "repeat_index": "0",
            "backend": "custom-cuda-paged",
            "num_requests": "1",
            "max_slots": "1",
            "max_new_tokens": "8",
            "block_size_tokens": "16",
            "total_kv_blocks": "256",
            "dtype": "float16",
            "device": "cuda",
            "prompt_set": "capitals",
            "total_wall_seconds": "1.0",
            "tokens_generated": "8",
            "tokens_per_second": "8.0",
            "decode_iterations": "8",
            "decode_batches_built": "8",
            "admitted_count": "1",
            "decode_stalls": "0",
            "kv_allocation_failures": "0",
            "kv_oom_evictions": "0",
            "late_admissions": "0",
            "early_finishes": "1",
            "backend_ms_median": "20.0",
            "backend_ms_p95": "25.0",
            "backend_ms_min": "19.0",
            "backend_ms_max": "26.0",
            "backend_ms_mean": "21.0",
            "kv_peak_used_blocks": "1",
            "kv_final_used_blocks": "0",
            "kv_final_free_blocks": "256",
            "kv_final_utilization": "0.0",
            "all_finished": "True",
            "generated_text_by_request": "{}",
            "expected_text_by_request": "{}",
            "correctness_passed": "True",
        },
        {
            "run_kind": "measured",
            "repeat_index": "0",
            "backend": "custom-cuda-paged",
            "num_requests": "2",
            "max_slots": "2",
            "max_new_tokens": "8",
            "block_size_tokens": "16",
            "total_kv_blocks": "256",
            "dtype": "float16",
            "device": "cuda",
            "prompt_set": "capitals",
            "total_wall_seconds": "1.0",
            "tokens_generated": "16",
            "tokens_per_second": "16.0",
            "decode_iterations": "8",
            "decode_batches_built": "8",
            "admitted_count": "2",
            "decode_stalls": "0",
            "kv_allocation_failures": "0",
            "kv_oom_evictions": "0",
            "late_admissions": "1",
            "early_finishes": "2",
            "backend_ms_median": "40.0",
            "backend_ms_p95": "45.0",
            "backend_ms_min": "39.0",
            "backend_ms_max": "46.0",
            "backend_ms_mean": "41.0",
            "kv_peak_used_blocks": "2",
            "kv_final_used_blocks": "0",
            "kv_final_free_blocks": "256",
            "kv_final_utilization": "0.0",
            "all_finished": "True",
            "generated_text_by_request": "{}",
            "expected_text_by_request": "{}",
            "correctness_passed": "True",
        },
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_runtime_benchmarks",
            "--csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--block-size-tokens",
            "16",
        ],
    )

    main()

    assert (output_dir / "tokens_per_second_vs_requests.png").exists()
    assert (output_dir / "backend_ms_median_vs_requests.png").exists()
    assert (output_dir / "backend_ms_p95_vs_requests.png").exists()
    assert (output_dir / "kv_peak_blocks_vs_requests.png").exists()
    assert (output_dir / "plot_summary.md").exists()