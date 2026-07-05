from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class MatrixScenario:
    """
    One benchmark scenario in the scheduler policy matrix.

    A scenario combines a workload shape with a slot configuration. Policies
    are applied across the same scenarios so benchmark rows are comparable.
    """

    name: str
    prompt_set: str
    num_requests: int
    max_slots: int
    max_new_tokens: int


@dataclass(frozen=True)
class MatrixPolicy:
    """
    One scheduler policy configuration in the benchmark matrix.
    """

    name: str
    scheduling_policy: str
    max_decode_batch_size: int


SCENARIOS = [
    MatrixScenario(
        name="capitals_control",
        prompt_set="capitals",
        num_requests=4,
        max_slots=4,
        max_new_tokens=8,
    ),
    MatrixScenario(
        name="mixed_no_pressure",
        prompt_set="mixed_short_long",
        num_requests=8,
        max_slots=8,
        max_new_tokens=16,
    ),
    MatrixScenario(
        name="mixed_slot_pressure",
        prompt_set="mixed_short_long",
        num_requests=16,
        max_slots=4,
        max_new_tokens=16,
    ),
]


POLICIES = [
    MatrixPolicy(
        name="fcfs",
        scheduling_policy="fcfs",
        max_decode_batch_size=4,
    ),
    MatrixPolicy(
        name="decode_budget_2",
        scheduling_policy="decode_budget",
        max_decode_batch_size=2,
    ),
]


def run_command(command: list[str]) -> None:
    """
    Run one benchmark command and fail immediately if it fails.
    """

    print()
    print("[run]", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", type=str, default="custom-cuda-paged")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-kv-blocks", type=int, default=256)
    parser.add_argument("--block-size-tokens", type=str, default="16")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks"),
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_dir = args.output_dir / f"policy_matrix_{timestamp}"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    print(f"[matrix] output_dir={matrix_dir}")

    for scenario in SCENARIOS:
        for policy in POLICIES:
            scenario_output_dir = matrix_dir / scenario.name / policy.name

            command = [
                sys.executable,
                "-m",
                "experiments.benchmarks.bench_runtime",
                "--backend",
                args.backend,
                "--prompt-set",
                scenario.prompt_set,
                "--scheduling-policy",
                policy.scheduling_policy,
                "--max-decode-batch-size",
                str(policy.max_decode_batch_size),
                "--num-requests",
                str(scenario.num_requests),
                "--max-slots",
                str(scenario.max_slots),
                "--max-new-tokens",
                str(scenario.max_new_tokens),
                "--block-size-tokens",
                args.block_size_tokens,
                "--total-kv-blocks",
                str(args.total_kv_blocks),
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--warmup-runs",
                str(args.warmup_runs),
                "--repeat-runs",
                str(args.repeat_runs),
                "--output-dir",
                str(scenario_output_dir),
            ]

            run_command(command)

    print()
    print(f"[done] wrote policy matrix artifacts under: {matrix_dir}")


if __name__ == "__main__":
    main()