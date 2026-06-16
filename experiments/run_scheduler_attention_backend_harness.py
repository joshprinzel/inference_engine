from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthetic_decode_engine import SyntheticDecodeConfig, SyntheticDecodeEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "reference"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--total-blocks", type=int, default=0)
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/scheduler_attention_backend_harness.csv",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    config = SyntheticDecodeConfig(
        backend=args.backend,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size_tokens=args.block_size,
        num_layers=args.num_layers,
        total_blocks=args.total_blocks if args.total_blocks > 0 else None,
        dtype="float16",
        device="cuda",
    )

    engine = SyntheticDecodeEngine(config)
    engine.initialize()
    result = engine.run()

    results_path = REPO_ROOT / args.results_csv
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "decode_step",
                "active_batch_size",
                "backend",
                "backend_ms",
                "kv_used_blocks",
                "kv_free_blocks",
                "kv_utilization",
                "total_tokens_emitted",
            ],
        )
        writer.writeheader()
        for metric in result.step_metrics:
            writer.writerow(asdict(metric))

    print("scheduler attention backend harness")
    print("-----------------------------------")
    print(f"backend:              {args.backend}")
    print(f"batch_size:           {args.batch_size}")
    print(f"prompt_tokens:        {args.prompt_tokens}")
    print(f"max_new_tokens:       {args.max_new_tokens}")
    print(f"decode_steps:         {result.decode_steps}")
    print(f"total_tokens_emitted: {result.total_tokens_emitted}")
    print(f"wall_seconds:         {result.wall_seconds:.6f}")
    print(f"tokens_per_second:    {result.tokens_per_second:.2f}")
    print(f"backend_med_ms:       {result.backend_med_ms:.6f}")
    print(f"backend_min_ms:       {result.backend_min_ms:.6f}")
    print(f"backend_max_ms:       {result.backend_max_ms:.6f}")
    print(f"results_csv:          {results_path}")
    print("final_block_manager:")
    print(result.final_block_manager_snapshot)


if __name__ == "__main__":
    main()