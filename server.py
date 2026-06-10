from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from metrics_store import MetricsStore
from model_runner import ModelRunner
from request_queue import RequestQueue
from request_state import RequestState
from scheduler import Scheduler
from schemas import GenerateRequest, GenerateResponse
from continuous_scheduler import ContinuousScheduler
from kv_block_manager import KVBlockManager

app = FastAPI(title="Toy LLM Inference Server")

runner = ModelRunner()
request_queue = RequestQueue()
metrics_store = MetricsStore()
kv_block_manager = KVBlockManager(
    total_blocks=128,
    block_size_tokens=16
)

# Normal Scheduluer from previous iteration
# scheduler = Scheduler(
#     runner=runner,
#     request_queue=request_queue,
#     metrics_store=metrics_store,
#     max_batch_size=4,
#     batch_wait_seconds=0.005
# )


# New Scheduler with Continous Scheduling semantics
scheduler = ContinuousScheduler(
    runner=runner,
    request_queue=request_queue,
    metrics_store=metrics_store,
    kv_block_manager=kv_block_manager,
    max_slots=4
)




@app.on_event("startup")
def startup() -> None:
    scheduler.start()


@app.get("/health")
def health() -> dict:
    engine_snapshot = scheduler.snapshot()

    return {
        "status": "ok",
        "model_name": runner.model_name,
        "device": runner.device,
        "queued_requests": request_queue.qsize(),
        "engine": engine_snapshot,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
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
    snapshot["engine"] = scheduler.snapshot()
    return snapshot