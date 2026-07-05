from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadRequestSpec:
    """
    Static request specification used by benchmark workloads.

    This object describes what should be enqueued into the runtime. It does 
    not own scheduler state, model tensor, KV blocks, timing fields, or 
    generated output. Those belong to RequestState and the runtime itself.

    max_new_tokens is per-request so benchmark workloads can model mixed
    decode lengths without changing the scheduler API.
    """

    request_id: str
    prompt: str
    expected_prefix: str
    max_new_tokens: int


CAPITAL_PROMPTS = {
    "france": "The capital of France is",
    "germany": "The capital of Germany is",
    "italy": "The capital of Italy is",
    "spain": "The capital of Spain is",
}

CAPITAL_EXPECTED_PREFIXES = {
    "france": "Paris",
    "germany": "Berlin",
    "italy": "Rome",
    "spain": "Madrid",
}



def build_capitals_workload(
        *,
        num_requests: int,
        max_new_tokens: int,
) -> list[WorkloadRequestSpec]:
    """
    Builds the deterministic correctness/control workload

    Every request uses a simple capital-city completion prompt and the same
    decode length. This workload is useful for quick regression checks because
    expected prefixes are stable and easy to validate
    """

    if num_requests <= 0:
        raise ValueError("num_request must be positive")
    if max_new_tokens <=0:
        raise ValueError("max_new_tokens must be positive")
    
    items = list(CAPITAL_PROMPTS.items())
    specs: list[WorkloadRequestSpec] = []

    for index in range(num_requests):
        name, prompt = items[index % len(items)]
        specs.append(
            WorkloadRequestSpec(
                request_id=f"req-{index}-{name}",
                prompt=prompt,
                expected_prefix=CAPITAL_EXPECTED_PREFIXES[name],
                max_new_tokens=max_new_tokens,
            )
        )
    return specs


def build_mixed_short_long_workload(
        *,
        num_requests: int,
        max_new_tokens: int
) -> list[WorkloadRequestSpec]:
    """
    Build a deterministic mixed-length decode workload

    This workload keeps prompts correctness-checkable while varying generation
    length across requests. It is useful for evaluating scheduler behavior when 
    requests finish at different times and active decode batches shrink over the run.

    The workload intentionally does not simulate staggered requests. Slot
    pressure should be created by running with num_requests > max_slots
    """

    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    
    items = list(CAPITAL_PROMPTS.items())
    specs: list[WorkloadRequestSpec] = []

    token_pattern = [
        max(1, max_new_tokens // 4),
        max(1, max_new_tokens // 2),
        max_new_tokens,
        max_new_tokens
    ]

    for index in range(num_requests):
        name, prompt = items[index % len(items)]
        request_max_new_tokens = token_pattern[index % len(token_pattern)]

        specs.append(
            WorkloadRequestSpec(
                request_id=f"req-{index}-{name}-new{request_max_new_tokens}",
                prompt=prompt,
                expected_prefix=CAPITAL_EXPECTED_PREFIXES[name],
                max_new_tokens=request_max_new_tokens
            )
        )
    
    return specs

def build_workload(
    *,
    prompt_set: str,
    num_requests: int,
    max_new_tokens: int,
) -> list[WorkloadRequestSpec]:
    """
    Build benchmark request specs for a named workload.

    Supported workloads:
      capitals:
        Fixed-length correctness/control workload.

      mixed_short_long:
        Mixed generation-length workload for scheduler behavior experiments.
        Use num_requests > max_slots in the benchmark CLI to create slot
        pressure with this workload.
    """

    if prompt_set == "capitals":
        return build_capitals_workload(
            num_requests=num_requests,
            max_new_tokens=max_new_tokens,
        )

    if prompt_set == "mixed_short_long":
        return build_mixed_short_long_workload(
            num_requests=num_requests,
            max_new_tokens=max_new_tokens,
        )

    raise ValueError(f"Unsupported prompt_set: {prompt_set}")