import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from runtime.engine_scheduler import EngineScheduler
from runtime.kv_block_manager import KVBlockManager
from runtime.metrics_store import MetricsStore
from runtime.request_queue import RequestQueue
from runtime.request_state import RequestState

from engines.model_runner import ModelRunner
from schemas import GenerateRequest, GenerateResponse

from engines.hf_decode_engine import HFDecodeEngine
from engines.synthetic_cuda_decode_engine import SyntheticCudaDecodeEngine


ENGINE_BACKEND = os.getenv("ENGINE_BACKEND", "hf").lower()

MAX_SLOTS = int(os.getenv("MAX_SLOTS", "4"))
KV_TOTAL_BLOCKS = int(os.getenv("KV_TOTAL_BLOCKS", "512"))
KV_BLOCK_SIZE_TOKENS = int(os.getenv("KV_BLOCK_SIZE_TOKENS", "16"))

SYNTHETIC_NUM_QUERY_HEADS = int(os.getenv("SYNTHETIC_NUM_QUERY_HEADS", "16"))
SYNTHETIC_NUM_KV_HEADS = int(os.getenv("SYNTHETIC_NUM_KV_HEADS", "4"))
SYNTHETIC_HEAD_DIM = int(os.getenv("SYNTHETIC_HEAD_DIM", "128"))
SYNTHETIC_DEVICE = os.getenv("SYNTHETIC_DEVICE", "cuda")
SYNTHETIC_ATTENTION_BACKEND = os.getenv("SYNTHETIC_ATTENTION_BACKEND", "cuda")


app = FastAPI(title="Toy LLM Inference Server")

request_queue = RequestQueue()
metrics_store = MetricsStore()

kv_block_manager = KVBlockManager(
    total_blocks=KV_TOTAL_BLOCKS,
    block_size_tokens=KV_BLOCK_SIZE_TOKENS,
)

runner: ModelRunner | None = None


def build_decode_engine():
    global runner

    if ENGINE_BACKEND == "hf":
        runner = ModelRunner()
        return HFDecodeEngine(runner)

    if ENGINE_BACKEND == "synthetic-cuda":
        runner = None
        return SyntheticCudaDecodeEngine(
            total_blocks=KV_TOTAL_BLOCKS,
            block_size_tokens=KV_BLOCK_SIZE_TOKENS,
            num_layers=1,
            num_query_heads=SYNTHETIC_NUM_QUERY_HEADS,
            num_kv_heads=SYNTHETIC_NUM_KV_HEADS,
            head_dim=SYNTHETIC_HEAD_DIM,
            dtype="float16",
            device=SYNTHETIC_DEVICE,
            attention_backend=SYNTHETIC_ATTENTION_BACKEND,
        )

    raise ValueError(
        f"Unknown ENGINE_BACKEND={ENGINE_BACKEND!r}. "
        "Expected 'hf' or 'synthetic-cuda'."
    )


decode_engine = build_decode_engine()

scheduler = EngineScheduler(
    decode_engine=decode_engine,
    request_queue=request_queue,
    metrics_store=metrics_store,
    kv_block_manager=kv_block_manager,
    max_slots=MAX_SLOTS,
)


@app.on_event("startup")
def startup() -> None:
    scheduler.start()


@app.on_event("shutdown")
def shutdown() -> None:
    scheduler.stop()


@app.get("/health")
def health() -> dict:
    engine_snapshot = scheduler.snapshot()

    return {
        "status": "ok",
        "engine_backend": ENGINE_BACKEND,
        "model_name": runner.model_name if runner is not None else "synthetic-cuda",
        "device": decode_engine.device,
        "queued_requests": request_queue.qsize(),
        "engine": engine_snapshot,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if runner is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "/generate is only available for ENGINE_BACKEND=hf. "
                "Use /generate_stream for synthetic-cuda."
            ),
        )

    result = runner.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )
    return GenerateResponse(**result)


@app.post("/generate_stream")
def generate_stream(request: GenerateRequest):
    request_state = RequestState(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    request_queue.put(request_state)

    def token_generator():
        while True:
            item = request_state.output_queue.get()

            if item is None:
                break

            yield item

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )


@app.get("/metrics_json")
def metrics_json() -> dict:
    snapshot = metrics_store.snapshot()
    snapshot["queued_requests"] = request_queue.qsize()
    snapshot["engine_backend"] = ENGINE_BACKEND
    snapshot["engine"] = scheduler.snapshot()
    return snapshot