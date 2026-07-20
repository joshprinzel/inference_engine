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
from runtime.scheduling_policy import SchedulingPolicy

@dataclass(frozen=True)
class PlaygroundResult:
    prompt: str
    generated_text: str
    full_text: str

    prompt_tokens: int
    max_new_tokens: int
    tokens_generated: int

    block_size_tokens: int
    total_kv_blocks: int
    max_slots: int
    policy_name: str

    queue_wait_ms: float | None
    ttft_ms: float | None
    decode_latency_ms: float | None
    latency_ms: float | None

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
    step_trace: list[dict]

@dataclass(frozen=True)
class MultiPromptRequestResult:
    request_id: str
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    generated_tokens: int
    final_status: str
    error: str | None
    queue_wait_ms: float | None
    ttft_ms: float | None
    decode_latency_ms: float | None
    latency_ms: float | None


@dataclass(frozen=True)
class MultiPromptPlaygroundResult:
    request_results: list[MultiPromptRequestResult]
    max_new_tokens: int
    block_size_tokens: int
    total_kv_blocks: int
    max_slots: int
    policy_name: str
    tokens_generated: int
    tokens_per_second: float
    total_wall_seconds: float

    avg_queue_wait_ms: float | None
    avg_ttft_ms: float | None
    avg_decode_latency_ms: float | None
    avg_latency_ms: float | None

    decode_iterations: int
    decode_batches_built: int
    backend_ms_median: float
    backend_ms_p95: float
    backend_ms_min: float
    backend_ms_max: float
    kv_peak_used_blocks: int
    kv_final_used_blocks: int
    kv_final_free_blocks: int
    all_finished: bool
    last_decode_batch: dict
    step_trace: list[dict]



def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    sorted_values = sorted(values)
    index = int(round((p/100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]

def mean_optional(values: list[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return sum(present_values) / len(present_values)

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
    scheduling_policy: SchedulingPolicy | None = None
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
        scheduling_policy=scheduling_policy,
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

    step_trace: list[dict] = []

    for step_index in range(max_scheduler_steps):
        scheduler.step()

        snapshot = scheduler.snapshot()

        last_decode_batch = snapshot.get("last_decode_batch") or {}
        batch_size = int(last_decode_batch.get("num_requests",0) or 0)

        step_trace.append(
            {
                "step": step_index + 1,
                "waiting": int(snapshot["waiting"]),
                "active": int(snapshot["active"]),
                "finished": int(snapshot["finished"]),
                "active_prefill_tokens_remaining": int(snapshot["active_prefill_tokens_remaining"]),
                "active_decode_tokens_remaining": int(snapshot["active_decode_tokens_remaining"]),
                "active_estimated_tokens_remaining": int(snapshot["active_estimated_tokens_remaining"]),
                "waiting_prefill_tokens_remaining": int(snapshot["waiting_prefill_tokens_remaining"]),
                "waiting_decode_tokens_remaining": int(snapshot["waiting_decode_tokens_remaining"]),
                "waiting_estimated_tokens_remaining": int(snapshot["waiting_estimated_tokens_remaining"]),
                "tokens_generated": int(snapshot["tokens_generated"]),
                "decode_iterations": int(snapshot["decode_iterations"]),
                "decode_batches_built": int(snapshot["decode_batches_built"]),
                "decode_batch_size": batch_size,
                "kv_used_blocks": int(snapshot["kv_used_blocks"]),
                "kv_free_blocks": int(snapshot["kv_free_blocks"]),
                "last_backend_ms": (
                    float(scheduler.last_backend_ms)
                    if scheduler.last_backend_ms is not None
                    else 0.0
                ),
            }
        )
        if scheduler.last_backend_ms is not None:
            backend_ms_values.append(float(scheduler.last_backend_ms))
        
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

        prompt_tokens=int(finished_request.prompt_tokens),
        max_new_tokens=max_new_tokens,
        tokens_generated=tokens_generated,

        block_size_tokens=block_size_tokens,
        total_kv_blocks=total_kv_blocks,
        max_slots=max_slots,
        policy_name=str(final_snapshot["policy_name"]),

        queue_wait_ms=finished_request.queue_wait_ms,
        ttft_ms=finished_request.ttft_ms,
        decode_latency_ms=finished_request.decode_latency_ms,
        latency_ms=finished_request.latency_ms,

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
        step_trace=step_trace,
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


@torch.inference_mode()
def run_tinyllama_multi_prompt_with_engine(
    *,
    engine: CustomLlamaDecodeEngine,
    prompts: list[str],
    max_new_tokens: int,
    block_size_tokens: int,
    total_kv_blocks: int,
    max_slots: int,
    scheduling_policy: SchedulingPolicy | None = None,
    device: str,
) -> MultiPromptPlaygroundResult:
    if not prompts:
        raise ValueError("prompts must not be empty")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is not available")
    
    if device == "cuda":
        torch.cuda.synchronize()

    engine.kv_cache_pool.zero_()

    request_queue = RequestQueue()
    metrics_store = MetricsStore()
    kv_block_manager = KVBlockManager(
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens
    )

    scheduler = EngineScheduler(
        decode_engine=engine,
        request_queue=request_queue,
        metrics_store=metrics_store,
        kv_block_manager=kv_block_manager,
        max_slots=max_slots,
        scheduling_policy=scheduling_policy
    )

    for index, prompt in enumerate(prompts):
        request_state = RequestState(
            request_id=f"playground-request-{index}",
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        request_queue.put(request_state)
    
    backend_ms_values: list[float] = []
    kv_used_blocks_values: list[int] = []
    step_trace: list[dict] = []

    max_scheduler_steps = max_new_tokens + len(prompts) + max_slots + 16

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for step_index in range(max_scheduler_steps):
        scheduler.step()

        snapshot = scheduler.snapshot()
        last_decode_batch = snapshot.get("last_decode_batch") or {}
        batch_size = int(last_decode_batch.get("num_requests",0) or 0)

        step_trace.append(
            {
                "step": step_index + 1,
                "waiting": int(snapshot["waiting"]),
                "active": int(snapshot["active"]),
                "finished": int(snapshot["finished"]),
                "active_prefill_tokens_remaining": int(snapshot["active_prefill_tokens_remaining"]),
                "active_decode_tokens_remaining": int(snapshot["active_decode_tokens_remaining"]),
                "active_estimated_tokens_remaining": int(snapshot["active_estimated_tokens_remaining"]),
                "waiting_prefill_tokens_remaining": int(snapshot["waiting_prefill_tokens_remaining"]),
                "waiting_decode_tokens_remaining": int(snapshot["waiting_decode_tokens_remaining"]),
                "waiting_estimated_tokens_remaining": int(snapshot["waiting_estimated_tokens_remaining"]),
                "tokens_generated": int(snapshot["tokens_generated"]),
                "decode_iterations": int(snapshot["decode_iterations"]),
                "decode_batches_built": int(snapshot["decode_batches_built"]),
                "decode_batch_size": batch_size,
                "kv_used_blocks": int(snapshot["kv_used_blocks"]),
                "kv_free_blocks": int(snapshot["kv_free_blocks"]),
                "last_backend_ms": (
                    float(scheduler.last_backend_ms)
                    if scheduler.last_backend_ms is not None
                    else 0.0
                ),
            }
        )

        if scheduler.last_backend_ms is not None:
            backend_ms_values.append(float(scheduler.last_backend_ms))

        kv_used_blocks_values.append(int(snapshot["kv_used_blocks"]))

        if len(scheduler.finished) >= len(prompts):
            break

    if device == "cuda":
        torch.cuda.synchronize()

    total_wall_seconds = time.perf_counter() - start

    final_snapshot = scheduler.snapshot()
    kv_snapshot = scheduler.kv_block_manager.snapshot()

    finished_by_id = {
        request.request_id: request
        for request in scheduler.finished
    }

    request_results: list[MultiPromptRequestResult] = []

    for index, prompt in enumerate(prompts):
        request_id = f"playground-request-{index}"
        finished_request = finished_by_id.get(request_id)

        if finished_request is None:
            request_results.append(
                MultiPromptRequestResult(
                    request_id=request_id,
                    prompt=prompt,
                    generated_text="",
                    full_text=prompt,
                    prompt_tokens=0,
                    generated_tokens=0,
                    final_status="not_finished",
                    error="Request did not finish within scheduler step limit",
                    queue_wait_ms=None,
                    ttft_ms=None,
                    decode_latency_ms=None,
                    latency_ms=None
                )
            )
            continue

        generated_text = finished_request.generated_text

        request_results.append(
            MultiPromptRequestResult(
                request_id=request_id,
                prompt=prompt,
                generated_text=generated_text,
                full_text=prompt + generated_text,
                prompt_tokens=int(finished_request.prompt_tokens),
                generated_tokens=int(finished_request.generated_tokens),
                final_status=finished_request.status,
                error=repr(finished_request.error) if finished_request.error else None,
                queue_wait_ms=finished_request.queue_wait_ms,
                ttft_ms=finished_request.ttft_ms,
                decode_latency_ms=finished_request.decode_latency_ms,
                latency_ms=finished_request.latency_ms
            )
        )

    
    tokens_generated = int(scheduler.tokens_generated)
    tokens_per_second = (tokens_generated / total_wall_seconds if total_wall_seconds > 0 else 0.0)

    avg_queue_wait_ms = mean_optional([request.queue_wait_ms for request in request_results])
    avg_ttft_ms = mean_optional([request.ttft_ms for request in request_results])
    avg_decode_latency_ms = mean_optional([request.decode_latency_ms for request in request_results])
    avg_latency_ms = mean_optional([request.latency_ms for request in request_results])



    return MultiPromptPlaygroundResult(
        request_results=request_results,
        max_new_tokens=max_new_tokens,
        block_size_tokens=block_size_tokens,
        total_kv_blocks=total_kv_blocks,
        max_slots=max_slots,
        policy_name=str(final_snapshot["policy_name"]),
        tokens_generated=tokens_generated,
        tokens_per_second=tokens_per_second,
        total_wall_seconds=total_wall_seconds,

        avg_queue_wait_ms=avg_queue_wait_ms,
        avg_ttft_ms=avg_ttft_ms,
        avg_decode_latency_ms=avg_decode_latency_ms,
        avg_latency_ms=avg_latency_ms,
        
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
        all_finished=len(scheduler.finished) == len(prompts),
        last_decode_batch=final_snapshot.get("last_decode_batch") or {},
        step_trace=step_trace,
    )
    
