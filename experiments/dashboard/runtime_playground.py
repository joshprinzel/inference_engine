from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from engines.llama.custom_llama_decode_engine import CustomLlamaDecodeEngine
from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState

@dataclass(frozen=True)
class PlaygroundResult:
    prompt:str
    generated_text:str
    full_text:str
    max_new_tokens: int
    tokens_generated: int
    tokens_per_second: float
    total_wall_seconds: float
    decode_iterations: int
    decode_batches_built: int
    backend_ms_median: float
    backend_ms_p95: float
    backend_ms_min: float
    backend_ms_max: float
    kv_peak_used_blocks: int
    kv_final_used_blocks: int
    kv_final_free_blocks: int
    final_status: str
    error: str | None
    last_decode_batch: dict



def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    sorted_values = sorted(values)
    index = int(round((p/100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]

def create_playground_engine(
        *,
        block_size_tokens: int,
        total_kv_blocks: int,
        dtype: torch.dtype,
        device: str,
        attention_backend_name: str = "cuda"
) -> CustomLlamaDecodeEngine:
    return CustomLlamaDecodeEngine(
        device=device,
        dtype=dtype,
        total_kv_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
        attention_backend_name=attention_backend_name
    )


@torch.inference_mode()
def run_tinyllama_request_with_engine(
    *,
    engine: CustomLlamaDecodeEngine,
    prompt: str,
    max_new_tokens: int,
    block_size_tokens: int,
    total_kv_blocks: int,
    max_slots: int,
    device: str,
) -> PlaygroundResult:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is not available")

    if device == "cuda":
        torch.cuda.synchronize()

    # Important: reset physical KV tensors between dashboard runs.
    engine.kv_cache_pool.zero_()

    request_queue = RequestQueue()
    metrics_store = MetricsStore()
    kv_block_manager = KVBlockManager(
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
    )

    scheduler = EngineScheduler(
        decode_engine=engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=max_slots,
    )

    request_state = RequestState(
        request_id="playground-request",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    request_queue.put(request_state)

    backend_ms_values: list[float] = []
    kv_used_blocks_values: list[int] = []

    max_scheduler_steps = max_new_tokens + max_slots + 16

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(max_scheduler_steps):
        scheduler.step()

        if scheduler.last_backend_ms is not None:
            backend_ms_values.append(float(scheduler.last_backend_ms))

        snapshot = scheduler.snapshot()
        kv_used_blocks_values.append(int(snapshot["kv_used_blocks"]))

        if len(scheduler.finished) >= 1:
            break

    if device == "cuda":
        torch.cuda.synchronize()

    total_wall_seconds = time.perf_counter() - start

    if len(scheduler.finished) < 1:
        raise RuntimeError(
            f"Request did not finish within {max_scheduler_steps} scheduler steps"
        )

    final_snapshot = scheduler.snapshot()
    kv_snapshot = scheduler.kv_block_manager.snapshot()

    finished_request = scheduler.finished[0]
    generated_text = finished_request.generated_text

    tokens_generated = int(scheduler.tokens_generated)
    tokens_per_second = (
        tokens_generated / total_wall_seconds if total_wall_seconds > 0 else 0.0
    )

    return PlaygroundResult(
        prompt=prompt,
        generated_text=generated_text,
        full_text=prompt + generated_text,
        max_new_tokens=max_new_tokens,
        tokens_generated=tokens_generated,
        tokens_per_second=tokens_per_second,
        total_wall_seconds=total_wall_seconds,
        decode_iterations=int(final_snapshot["decode_iterations"]),
        decode_batches_built=int(final_snapshot["decode_batches_built"]),
        backend_ms_median=statistics.median(backend_ms_values)
        if backend_ms_values
        else 0.0,
        backend_ms_p95=percentile(backend_ms_values, 95.0),
        backend_ms_min=min(backend_ms_values) if backend_ms_values else 0.0,
        backend_ms_max=max(backend_ms_values) if backend_ms_values else 0.0,
        kv_peak_used_blocks=max(kv_used_blocks_values) if kv_used_blocks_values else 0,
        kv_final_used_blocks=int(kv_snapshot["used_blocks"]),
        kv_final_free_blocks=int(kv_snapshot["free_blocks"]),
        final_status=finished_request.status,
        error=repr(finished_request.error) if finished_request.error else None,
        last_decode_batch=final_snapshot.get("last_decode_batch") or {},
    )

@torch.inference_mode()
def run_tinyllama_request(
    prompt:str,
    max_new_tokens:int,
    block_size_tokens:int,
    total_kv_blocks:int,
    max_slots:int,
    dtype:torch.dtype,
    device:str,
    attention_backend_name: str = "cuda"
) -> PlaygroundResult:
    
    engine = create_playground_engine(
        block_size_tokens=block_size_tokens,
        total_kv_blocks=total_kv_blocks,
        dtype=dtype,
        device=device,
        attention_backend_name=attention_backend_name
    )

    return run_tinyllama_request_with_engine(
        engine=engine,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_size_tokens=block_size_tokens,
        total_kv_blocks=total_kv_blocks,
        max_slots=max_slots,
        device=device
    )
    
