import time
from queue import Empty
from threading import Lock, Thread
from typing import Optional

from metrics_store import MetricsStore
from model_runner import ModelRunner
from request_queue import RequestQueue
from request_state import RequestState
from kv_block_manager import KVBlockManager, KVBlockAllocationError


class ContinuousScheduler:
    """
    Layer 1 continuous scheduling semantics.

    This scheduler models iteration-level serving:
    - new requests can be admitted into free slots between decode steps
    - finished requests free slots immediately
    - each occupied slot decodes one token per engine step

    This version uses per-request KV state instead of one shared batched KV tensor.
    That makes the semantics clean before Layer 2 KV-cache ownership
    """

    def __init__(
            self,
            runner: ModelRunner,
            request_queue: RequestQueue,
            metrics_store: MetricsStore,
            kv_block_manager: KVBlockManager,
            max_slots: int = 4,
            step_sleep_seconds: float = 0.0,
            idle_sleep_seconds: float = 0.01) -> None:
        self.runner = runner
        self.request_queue = request_queue
        self.metrics_store = metrics_store
        self.kv_block_manager = kv_block_manager

        self.max_slots = max_slots
        self.step_sleep_seconds = step_sleep_seconds
        self.idle_sleep_seconds = idle_sleep_seconds

        self.waiting: list[RequestState] = []
        self.finished: list[RequestState] = []
        self.slots: list[Optional[RequestState]] = [None for _ in range(max_slots)]

        self._lock = Lock()
        self._thread: Thread | None = None
        self._running = False

        self.engine_step = 0

        self.queue_length_history: list[int] = []
        self.occupied_slots_history: list[int] = []
        self.finished_count_history: list[int] = []

        self.admitted_count = 0
        self.decode_steps = 0
        self.tokens_generated = 0
        self.late_admissions = 0
        self.early_finishes = 0

        self.decode_stalls = 0
        self.kv_allocation_failures = 0
        self.kv_oom_evictions = 0

    

    #----------------------------------------------------------------------------------------
    # Worker Lifecycle
    #-----------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        self._running = False
    

    #-------------------------------------------------------------------------------------------
    # State Accessors
    #-------------------------------------------------------------------------------------------

    def occupied_slot_count(self) -> int:
        return sum(1 for request_state in self.slots if request_state is not None)
    
    def has_free_slot(self) -> bool:
        return self.occupied_slot_count() < self.max_slots
    

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "engine_step": self.engine_step,
                "scheduler_type": "continuous_slots",
                "waiting": len(self.waiting),
                "active": self.occupied_slot_count(),
                "finished": len(self.finished),
                "max_slots": self.max_slots,
                "kv_cache": self.kv_block_manager.snapshot(),
                "slots": [
                    None if request_state is None else{
                        "request_id": request_state.request_id,
                        "status": request_state.status,
                        "generated_tokens": request_state.generated_tokens,
                        "max_new_tokens": request_state.max_new_tokens,
                    }
                    for request_state in self.slots
                ],
                "admitted_count": self.admitted_count,
                "decode_iterations": self.decode_steps,
                "decode_stalls": self.decode_stalls,
                "kv_allocation_failures": self.kv_allocation_failures,
                "kv_oom_evictions": self.kv_oom_evictions,
                "late_admissions": self.late_admissions,
                "early_finishes": self.early_finishes,
                "queue_length_history_tail": self.queue_length_history[-20:],
                "occupied_slots_history_tail": self.occupied_slots_history[-20:],
                "finished_count_history_tail": self.finished_count_history[-20:]
            }
    

    # ----------------------------------------------------------------------------
    # Lifecycle: arrivals / admission / decode / finish / metrics
    # ----------------------------------------------------------------------------

    def drain_external_queue(self) -> None:
        while True:
            try:
                request_state = self.request_queue.get_nowait()
            except Empty:
                break

            request_state.status = "waiting"
            self.waiting.append(request_state)
    
    def first_free_slot_index(self) -> int | None:
        for index, request_state in enumerate(self.slots):
            if request_state is None:
                return index
        return None
    
    def admit_waiting_requests(self) -> None:
        while self.waiting and self.has_free_slot():
            slot_index = self.first_free_slot_index()

            if slot_index is None:
                return
            
            request_state = self.waiting.pop(0)
            request_state.mark_admitted()

            print(
                f"admitting request_id={request_state.request_id} "
                f"max_new_tokens={request_state.max_new_tokens}",
                flush=True
            )

            try:
                prompt_tokens = self.runner.count_prompt_tokens(request_state.prompt)
                reserved_tokens = prompt_tokens + request_state.max_new_tokens

                if not self.kv_block_manager.can_allocate_tokens(reserved_tokens):
                    self.waiting.insert(0,request_state)
                    return
                
                request_id = str(request_state.request_id)
                block_table = self.kv_block_manager.allocate_for_tokens(
                    request_id=request_id,
                    num_tokens=request_state.prompt_tokens
                )
                
                request_state.block_table = block_table
                request_state.prompt_tokens = prompt_tokens

                self.runner.init_request_state(request_state)
                request_state.num_computed_tokens = request_state.prompt_tokens                
            except (KVBlockAllocationError, Exception) as error:
                self.kv_block_manager.free(str(request_state.request_id))
                request_state.mark_failed(error)
                self.finished.append(request_state)
                self.metrics_store.record_finished(request_state)
                continue

            #If other requests are already running, this is a late admission
            if self.occupied_slot_count() > 0:
                self.late_admissions += 1
            
            self.slots[slot_index] = request_state
            self.admitted_count += 1

    def decode_active_requests(self) -> None:
        occupied_indices = [
            index 
            for index, request_state in enumerate(self.slots)
            if request_state is not None
        ]
        if not occupied_indices:
            return
        
        self.decode_steps += 1

        decoded_any = False
        stalled_indices: list[int] = []

        for index in occupied_indices:
            request_state = self.slots[index]

            if request_state is None:
                continue
            if request_state.status == "finished":
                continue

            try:
                request_id = str(request_state.request_id)
                token_position = request_state.prompt_tokens + request_state.generated_tokens

                try:
                    self.kv_block_manager.ensure_capacity_for_token(
                        request_id=request_id,
                        token_position=token_position
                    )
                except KVBlockAllocationError:
                    self.decode_stalls += 1
                    self.kv_allocation_failures += 1
                    stalled_indices.append(index)
                    continue

                request_state.block_table = self.kv_block_manager.get_block_tables(request_id)
                text = self.runner.decode_one_token(request_state)
                if text:
                    request_state.append_text(text)
                else:
                    request_state.mark_first_token()
                
                request_state.generated_tokens += 1
                self.tokens_generated += 1

                request_state.num_computed_tokens = (request_state.prompt_tokens + request_state.generated_tokens)

                decoded_any = True

                if request_state.is_finished():
                    print(
                        f"finished request_id={request_state.request_id} "
                        f"generated_tokens={request_state.generated_tokens} "
                        f"max_new_tokens={request_state.max_new_tokens}",
                        flush=True
                    )
                    self.kv_block_manager.free(str(request_state.request_id))
                    request_state.mark_finished()
                    self.finished.append(request_state)
                    self.metrics_store.record_finished(request_state)

                    self.slots[index] = None
                    self.early_finishes += 1
            
            except Exception as error:
                self.kv_block_manager.free(str(request_state.request_id))
                request_state.mark_failed(error)
                self.finished.append(request_state)
                self.metrics_store.record_finished(request_state)
                self.slots[index] = None
        
        if not decoded_any and stalled_indices:
            victim_index = stalled_indices[-1]
            victim = self.slots[victim_index]

            if victim is not None:
                self.kv_block_manager.free(str(victim.request_id))

                victim.mark_failed(
                    RuntimeError(
                        "KV cache exhausted: request evicted to break decode-time "
                        "memory deadlock"

                    )
                )

                self.finished.append(victim)
                self.metrics_store.record_finished(victim)

                self.slots[victim_index] = None
                self.kv_oom_evictions += 1
    

    def record_history(self) -> None:
        self.queue_length_history.append(len(self.waiting))
        self.occupied_slots_history.append(self.occupied_slot_count())
        self.finished_count_history.append(len(self.finished))

    def step(self) -> None:
        with self._lock:
            self.drain_external_queue()
            self.admit_waiting_requests()
            self.decode_active_requests()
            self.record_history()
            self.engine_step += 1

    def _run_loop(self) -> None:
        while self._running:
            self.step()

            if self.step_sleep_seconds > 0:
                time.sleep(self.step_sleep_seconds)
            elif not self.waiting and self.occupied_slot_count() == 0:
                time.sleep(self.idle_sleep_seconds)
