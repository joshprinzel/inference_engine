from __future__ import annotations

import pytest
import torch

from experiments.benchmarks.bench_runtime import BenchmarkConfig, run_single_benchmark


pytestmark = [pytest.mark.cuda, pytest.mark.llama, pytest.mark.slow]


def test_bench_runtime_custom_cuda_paged_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    config = BenchmarkConfig(
        backend="custom-cuda-paged",
        num_requests=1,
        max_slots=1,
        max_new_tokens=4,
        block_size_tokens=16,
        total_kv_blocks=64,
        dtype="float16",
        device="cuda",
        prompt_set="capitals",
    )

    result = run_single_benchmark(
        config=config,
        run_kind="measured",
        repeat_index=0,
    )

    print(f"tokens_per_second={result.tokens_per_second}")
    print(f"backend_ms_median={result.backend_ms_median}")
    print(f"backend_ms_p95={result.backend_ms_p95}")
    print(f"kv_peak_used_blocks={result.kv_peak_used_blocks}")
    print(f"generated_text_by_request={result.generated_text_by_request}")

    assert result.run_kind == "measured"
    assert result.repeat_index == 0

    assert result.backend == "custom-cuda-paged"
    assert result.num_requests == 1
    assert result.max_slots == 1
    assert result.max_new_tokens == 4
    assert result.block_size_tokens == 16
    assert result.total_kv_blocks == 64
    assert result.dtype == "float16"
    assert result.device == "cuda"
    assert result.prompt_set == "capitals"

    assert result.all_finished is True
    assert result.correctness_passed is True

    assert result.tokens_generated == 4
    assert result.tokens_per_second > 0.0

    assert result.decode_iterations == 4
    assert result.decode_batches_built == 4
    assert result.admitted_count == 1
    assert result.decode_stalls == 0
    assert result.kv_allocation_failures == 0
    assert result.kv_oom_evictions == 0

    assert result.backend_ms_median > 0.0
    assert result.backend_ms_p95 > 0.0
    assert result.backend_ms_min > 0.0
    assert result.backend_ms_max > 0.0
    assert result.backend_ms_mean > 0.0

    assert result.kv_peak_used_blocks >= 1
    assert result.kv_final_used_blocks == 0
    assert result.kv_final_free_blocks == 64
    assert result.kv_final_utilization == 0.0

    assert "req-0-france" in result.generated_text_by_request
    assert "Paris" in result.generated_text_by_request