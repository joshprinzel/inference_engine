import pytest

from experiments.benchmarks.workloads import build_workload


@pytest.mark.runtime
def test_capitals_workload_builds_expected_number_of_specs() -> None:
    specs = build_workload(
        prompt_set="capitals",
        num_requests=6,
        max_new_tokens=8,
    )

    assert len(specs) == 6
    assert all(spec.max_new_tokens == 8 for spec in specs)
    assert specs[0].request_id == "req-0-france"
    assert specs[0].expected_prefix == "Paris"


@pytest.mark.runtime
def test_mixed_short_long_workload_varies_decode_lengths() -> None:
    specs = build_workload(
        prompt_set="mixed_short_long",
        num_requests=8,
        max_new_tokens=16,
    )

    max_new_tokens_by_request = [spec.max_new_tokens for spec in specs]

    assert len(specs) == 8
    assert max_new_tokens_by_request == [4, 8, 16, 16, 4, 8, 16, 16]
    assert specs[0].request_id == "req-0-france-new4"
    assert specs[1].request_id == "req-1-germany-new8"


@pytest.mark.runtime
def test_workload_rejects_unknown_prompt_set() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt_set"):
        build_workload(
            prompt_set="unknown",
            num_requests=4,
            max_new_tokens=8,
        )


@pytest.mark.runtime
@pytest.mark.parametrize("num_requests,max_new_tokens", [(0, 8), (4, 0)])
def test_workload_rejects_invalid_sizes(
    num_requests: int,
    max_new_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        build_workload(
            prompt_set="capitals",
            num_requests=num_requests,
            max_new_tokens=max_new_tokens,
        )