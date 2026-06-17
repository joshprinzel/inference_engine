"""
Legacy standalone synthetic decode harness.

Current engine-agnostic CUDA path:
    EngineScheduler -> SyntheticCudaDecodeEngine -> DecodeBatch -> AttentionBackend
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from runtime.attention_backend import AttentionBackend, build_attention_backend
from runtime.decode_batch import DecodeBatch, build_decode_batch
from runtime.kv_block_manager import KVBlockManager
from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.request_state import RequestState


@dataclass
class SyntheticDecodeConfig:
    backend: str = "cuda"
    batch_size: int = 32
    prompt_tokens: int = 512
    max_new_tokens: int = 32
    num_query_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    block_size_tokens: int = 8
    num_layers: int = 1
    total_blocks: int | None = None
    dtype: str = "float16"
    device: str = "cuda"


@dataclass
class DecodeStepMetrics:
    decode_step: int
    active_batch_size: int
    backend: str
    backend_ms: float
    kv_used_blocks: int
    kv_free_blocks: int
    kv_utilization: float
    total_tokens_emitted: int


@dataclass
class SyntheticDecodeResult:
    decode_steps: int
    total_tokens_emitted: int
    wall_seconds: float
    backend_med_ms: float
    backend_min_ms: float
    backend_max_ms: float
    tokens_per_second: float
    step_metrics: list[DecodeStepMetrics]
    final_block_manager_snapshot: dict


class SyntheticDecodeEngine:
    """
    Scheduler-owned synthetic decode engine.

    This is not full model execution.

    It validates the runtime path:

        RequestState
          -> KVBlockManager
          -> DecodeBatch
          -> KVCachePool
          -> AttentionBackend
          -> CUDA paged attention kernel

    Query tensors, generated token IDs, and generated-token K/V writes are synthetic.
    """

    def __init__(self, config: SyntheticDecodeConfig) -> None:
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self.config = config
        self.layer_id = 0

        max_tokens_per_request = config.prompt_tokens + config.max_new_tokens
        blocks_per_request = (
            max_tokens_per_request + config.block_size_tokens - 1
        ) // config.block_size_tokens
        required_blocks = config.batch_size * blocks_per_request

        total_blocks = config.total_blocks
        if total_blocks is None:
            total_blocks = required_blocks

        if total_blocks < required_blocks:
            raise ValueError(
                f"total_blocks={total_blocks} is too small. "
                f"Need at least {required_blocks}."
            )

        self.layout = KVCacheLayout(
            num_layers=config.num_layers,
            total_blocks=total_blocks,
            block_size_tokens=config.block_size_tokens,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=config.device,
        )

        self.cache_pool = KVCachePool(self.layout)
        self.cache_pool.zero_()

        self.block_manager = KVBlockManager(
            total_blocks=total_blocks,
            block_size_tokens=config.block_size_tokens,
        )

        self.backend: AttentionBackend = build_attention_backend(config.backend)
        self.request_states = self._make_requests()

    def initialize(self) -> None:
        for request_state in self.request_states:
            block_table = self.block_manager.allocate_for_tokens(
                request_id=str(request_state.request_id),
                num_tokens=self.config.prompt_tokens,
            )
            request_state.block_table = block_table
            self._fill_prompt_kv(block_table)

    def run(self) -> SyntheticDecodeResult:
        wall_start = time.perf_counter()

        decode_step = 0
        total_tokens_emitted = 0
        step_metrics: list[DecodeStepMetrics] = []

        while True:
            active = [
                request_state
                for request_state in self.request_states
                if request_state.status != "finished"
            ]

            if not active:
                break

            decode_batch = build_decode_batch(
                request_states=active,
                kv_block_manager=self.block_manager,
                device=self.config.device,
            )

            if decode_batch.batch_size == 0:
                break

            q = self._make_synthetic_q(decode_batch.batch_size)

            _, backend_ms = self._time_backend_decode(
                q=q,
                decode_batch=decode_batch,
            )

            for request_state in active:
                token_position = (
                    request_state.prompt_tokens + request_state.generated_tokens
                )

                self.block_manager.ensure_capacity_for_token(
                    request_id=str(request_state.request_id),
                    token_position=token_position,
                )

                block_table = self.block_manager.get_block_tables(
                    str(request_state.request_id)
                )
                request_state.block_table = block_table

                self._write_generated_token_kv(
                    block_table=block_table,
                    token_position=token_position,
                )

                request_state.generated_tokens += 1
                total_tokens_emitted += 1

                request_state.next_token = torch.tensor(
                    [[1000 + request_state.generated_tokens]],
                    dtype=torch.int64,
                )

                if request_state.is_finished():
                    request_state.mark_finished()
                    self.block_manager.free(str(request_state.request_id))

            step_metrics.append(
                DecodeStepMetrics(
                    decode_step=decode_step,
                    active_batch_size=decode_batch.batch_size,
                    backend=self.config.backend,
                    backend_ms=backend_ms,
                    kv_used_blocks=self.block_manager.used_block_count(),
                    kv_free_blocks=self.block_manager.free_block_count(),
                    kv_utilization=self.block_manager.utilization(),
                    total_tokens_emitted=total_tokens_emitted,
                )
            )

            decode_step += 1

        wall_end = time.perf_counter()
        wall_seconds = wall_end - wall_start

        backend_times = torch.tensor(
            [metric.backend_ms for metric in step_metrics],
            dtype=torch.float32,
        )

        return SyntheticDecodeResult(
            decode_steps=decode_step,
            total_tokens_emitted=total_tokens_emitted,
            wall_seconds=wall_seconds,
            backend_med_ms=float(backend_times.median().item()),
            backend_min_ms=float(backend_times.min().item()),
            backend_max_ms=float(backend_times.max().item()),
            tokens_per_second=(
                total_tokens_emitted / wall_seconds if wall_seconds > 0 else 0.0
            ),
            step_metrics=step_metrics,
            final_block_manager_snapshot=self.block_manager.snapshot(),
        )

    def _make_requests(self) -> list[RequestState]:
        requests: list[RequestState] = []

        for i in range(self.config.batch_size):
            request = RequestState(
                prompt=f"synthetic request {i}",
                max_new_tokens=self.config.max_new_tokens,
                request_id=f"req-{i}",
            )
            request.prompt_tokens = self.config.prompt_tokens
            request.generated_tokens = 0
            request.next_token = torch.tensor([[100 + i]], dtype=torch.int64)
            request.mark_admitted()
            request.mark_decoding()
            requests.append(request)

        return requests

    def _fill_prompt_kv(self, block_table: list[int]) -> None:
        for token_position in range(self.config.prompt_tokens):
            key = torch.randn(
                self.config.num_kv_heads,
                self.config.head_dim,
                device=self.config.device,
                dtype=self.cache_pool.key_cache.dtype,
            )
            value = torch.randn_like(key)

            self.cache_pool.write_request_token(
                layer_id=self.layer_id,
                block_table=block_table,
                token_position=token_position,
                key=key,
                value=value,
            )

    def _write_generated_token_kv(
        self,
        block_table: list[int],
        token_position: int,
    ) -> None:
        key = torch.randn(
            self.config.num_kv_heads,
            self.config.head_dim,
            device=self.config.device,
            dtype=self.cache_pool.key_cache.dtype,
        )
        value = torch.randn_like(key)

        self.cache_pool.write_request_token(
            layer_id=self.layer_id,
            block_table=block_table,
            token_position=token_position,
            key=key,
            value=value,
        )

    def _make_synthetic_q(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.config.num_query_heads,
            self.config.head_dim,
            device=self.config.device,
            dtype=torch.float16,
        )

    def _time_backend_decode(
        self,
        q: torch.Tensor,
        decode_batch: DecodeBatch,
    ) -> tuple[torch.Tensor, float]:
        if self.config.backend == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            out = self.backend.decode(
                q=q,
                cache_pool=self.cache_pool,
                layer_id=self.layer_id,
                block_tables=decode_batch.block_tables,
                seq_lens=decode_batch.seq_lens,
            )
            end.record()

            torch.cuda.synchronize()
            return out, float(start.elapsed_time(end))

        t0 = time.perf_counter()
        out = self.backend.decode(
            q=q,
            cache_pool=self.cache_pool,
            layer_id=self.layer_id,
            block_tables=decode_batch.block_tables,
            seq_lens=decode_batch.seq_lens,
        )

        if self.config.device == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        return out, (t1 - t0) * 1000.0