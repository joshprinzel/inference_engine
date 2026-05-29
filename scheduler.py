from threading import Thread, Lock
import time
from queue import Empty

from metrics_store import MetricsStore
from model_runner import ModelRunner
from request_state import RequestState
from request_queue import RequestQueue

class Scheduler:
    def __init__(
            self,
            runner: ModelRunner,
            request_queue: RequestQueue,
            metrics_store: MetricsStore,
            max_batch_size: int = 1,
            step_sleep_seconds: float = 0.0,
            batch_wait_seconds: float = 0.005,
            ) -> None:
        self.runner = runner
        self.request_queue = request_queue
        self.metrics_store = metrics_store

        self.max_batch_size = max_batch_size
        self.step_sleep_seconds = step_sleep_seconds
        self.batch_wait_seconds = batch_wait_seconds

        self.waiting: list[RequestState] = []
        self.active: list[RequestState] = []
        self.finished: list[RequestState] = []
        self.batch_state: dict | None = None

        self._lock = Lock()
        self._thread: Thread | None = None
        self._running = False

        self.engine_step = 0

        self.batch_size_history: list[int] = []
        self.queue_length_history: list[int] = []
        self.finished_count_history: list[int] = []
        self.completed_batch_sizes: list[int] = []


    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        self._running = False

    #-------------------------------------------------------------
    # State Accessors
    #-------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "engine_step": self.engine_step,
                "waiting": len(self.waiting),
                "active": len(self.active),
                "finished": len(self.finished),
                "max_batch_size": self.max_batch_size,
                "batch_size_history_tail": self.batch_size_history[-20:],
                "queue_length_history_tail": self.queue_length_history[-20:],
                "finished_count_history_tail": self.finished_count_history[-20:],
                "completed_batch_sizes_tail": self.completed_batch_sizes[-20:]
            }

    def has_batch_capacity(self) -> bool:
        return len(self.active) < self.max_batch_size
    
    #----------------------------------------------------------------------
    # Lifecyle: arrivals / admission / decode / finish / metrics
    #----------------------------------------------------------------------

    def drain_external_queue(self) -> None:
        while True:
            try:
                request_state = self.request_queue.get_nowait()
            except Empty:
                break
            request_state.status = "waiting"
            self.waiting.append(request_state)
    
    def admit_waiting_requests(self) -> None:
        if self.active:
            return
        if not self.waiting:
            return
        
        if len(self.waiting) < self.max_batch_size:
            time.sleep(self.batch_wait_seconds)
            self.drain_external_queue()
        
        batch: list[RequestState] = []

        while self.waiting and len(batch) < self.max_batch_size:
            request_state = self.waiting.pop(0)
            request_state.mark_admitted()
            batch.append(request_state)
        
        try:
            self.batch_state = self.runner.init_batch_request_states(batch)
            self.active = batch
        
        except Exception as error:
            for request_state in batch:
                request_state.mark_failed(error)
                self.finished.append(request_state)
                self.metrics_store.record_finished(request_state)
            self.batch_state = None
            self.active = []
    
    def decode_active_requests(self) -> None:
        if not self.active:
            return
        if self.batch_state is None:
            return
        
        try:
            new_batch_state = self.runner.decode_one_token_batch(
                request_states=self.active,
                batch_state=self.batch_state
            )
            texts = new_batch_state["texts"]

            for request_state, text in zip(self.active,texts):
                if request_state.is_finished():
                    continue
                if text:
                    request_state.append_text(text)
                else:
                    request_state.mark_first_token()
                
                request_state.generated_tokens += 1
            
            self.batch_state = {
                "past_key_values": new_batch_state["past_key_values"],
                "next_tokens": new_batch_state["next_tokens"],
                "attention_mask": new_batch_state["attention_mask"],
            }
        
        except Exception as error:
            for request_state in self.active:
                request_state.mark_failed(error)
    
    def remove_finished_requests(self) -> None:
        if not self.active:
            return
        
        failed_requests = [
            request_state
            for request_state in self.active
            if request_state.status == "failed"
        ]
        if failed_requests:
            completed_batch_size = len(self.active)
            for request_state in self.active:
                if request_state.status != "failed":
                    request_state.mark_failed(
                        RuntimeError("Batch failed because one or more requests failed")
                    )
                self.finished.append(request_state)
                self.metrics_store.record_finished(request_state)
            self.completed_batch_sizes.append(completed_batch_size)
            self.active = []
            self.batch_state = None
            return
        
        if not all(request_state.is_finished() for request_state in self.active):
            return
        
        completed_batch_size = len(self.active)
        
        for request_state in self.active:
            request_state.mark_finished()
            self.finished.append(request_state)
            self.metrics_store.record_finished(request_state)
        self.completed_batch_sizes.append(completed_batch_size)
        self.active = []
        self.batch_state = None


    def record_history(self) -> None:
        self.batch_size_history.append(len(self.active))
        self.queue_length_history.append(len(self.waiting))
        self.finished_count_history.append(len(self.finished))

    def step(self) -> None:
        with self._lock:
            self.drain_external_queue()
            self.admit_waiting_requests()
            self.decode_active_requests()
            self.remove_finished_requests()
            self.record_history()
            self.engine_step += 1


    def _run_loop(self) -> None:
        while self._running:
            self.step()

            if self.step_sleep_seconds > 0:
                time.sleep(self.step_sleep_seconds)
            elif not self.waiting and not self.active:
                time.sleep(0.001)
    