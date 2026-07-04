import time
from queue import Empty
from threading import Lock, Thread
from typing import Optional



from .metrics_store import MetricsStore
from .decode_engine import DecodeEngine, DecodeStepOutput
from .kv_block_manager import KVBlockAllocationError, KVBlockManager
from .request_queue import RequestQueue
from .request_state import RequestState
from .scheduling_policy import FCFSPolicy, SchedulingPolicy, DecodeBudgetPolicy


class EngineScheduler:
    """
    Engine-agnostic continuous scheduler.

    This scheduler owns:
        - request queue draining
        - slot admission
        - active request tracking
        - KV block reservation/free policy
        - engine step loop
        - metrics and snapshots

    It does NOT own:
        - Hugging Face execution
        - CUDA attention execution
        - model-specific token generation
        - backend-specific KV behavior

    Those belong to DecodeEngine implementations:
        - HFDecodeEngine
        - SyntheticCudaDecodeEngine
        - FutureCustomModelDecodeEngine
    """

    def __init__(
        self,
        decode_engine: DecodeEngine,
        request_queue: RequestQueue,
        metrics_store: MetricsStore,
        kv_block_manager: KVBlockManager,
        max_slots: int = 4,
        scheduling_policy: SchedulingPolicy | None = None,
        step_sleep_seconds: float = 0.0,
        idle_sleep_seconds: float = 0.01,
    ) -> None:
        self.decode_engine = decode_engine
        self.request_queue = request_queue
        self.metrics_store = metrics_store
        self.kv_block_manager = kv_block_manager
        self.scheduling_policy = scheduling_policy or FCFSPolicy()

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

        self.decode_batches_built = 0
        self.last_decode_batch_snapshot: dict | None = None
        self.last_backend_ms: float | None = None

    # -------------------------------------------------------------------------
    # Worker lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # -------------------------------------------------------------------------
    # State helpers
    # -------------------------------------------------------------------------

    def occupied_slot_count(self) -> int:
        return sum(1 for request_state in self.slots if request_state is not None)

    def has_free_slot(self) -> bool:
        return self.occupied_slot_count() < self.max_slots

    def first_free_slot_index(self) -> int | None:
        for index, request_state in enumerate(self.slots):
            if request_state is None:
                return index
        return None

    def active_request_states(self) -> list[RequestState]:
        return [
            request_state
            for request_state in self.slots
            if request_state is not None and request_state.status != "finished"
        ]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "engine_step": self.engine_step,
                "scheduler_type": "engine_agnostic_continuous_slots",
                "policy_name": self.scheduling_policy.name,
                "engine": type(self.decode_engine).__name__,
                "waiting": len(self.waiting),
                "active": self.occupied_slot_count(),
                "finished": len(self.finished),
                "max_slots": self.max_slots,
                "kv_cache": self.kv_block_manager.snapshot(),
                "kv_used_blocks": self.kv_block_manager.used_block_count(),
                "kv_free_blocks": self.kv_block_manager.free_block_count(),
                "kv_utilization": self.kv_block_manager.utilization(),
                "slots": [
                    None
                    if request_state is None
                    else {
                        "request_id": request_state.request_id,
                        "status": request_state.status,
                        "prompt_tokens": request_state.prompt_tokens,
                        "generated_tokens": request_state.generated_tokens,
                        "max_new_tokens": request_state.max_new_tokens,
                        "num_computed_tokens": request_state.num_computed_tokens,
                    }
                    for request_state in self.slots
                ],
                "admitted_count": self.admitted_count,
                "decode_iterations": self.decode_steps,
                "tokens_generated": self.tokens_generated,
                "decode_stalls": self.decode_stalls,
                "kv_allocation_failures": self.kv_allocation_failures,
                "kv_oom_evictions": self.kv_oom_evictions,
                "decode_batches_built": self.decode_batches_built,
                "last_decode_batch": self.last_decode_batch_snapshot,
                "last_backend_ms": self.last_backend_ms,
                "late_admissions": self.late_admissions,
                "early_finishes": self.early_finishes,
                "queue_length_history_tail": self.queue_length_history[-20:],
                "occupied_slots_history_tail": self.occupied_slots_history[-20:],
                "finished_count_history_tail": self.finished_count_history[-20:],
            }

    # -------------------------------------------------------------------------
    # Queue / admission
    # -------------------------------------------------------------------------

    def drain_external_queue(self) -> None:
        while True:
            try:
                request_state = self.request_queue.get_nowait()
            except Empty:
                break

            request_state.status = "waiting"
            self.waiting.append(request_state)
    
    def remove_from_waiting(self, request_state: RequestState) -> bool:
        for index, waiting_request in enumerate(self.waiting):
            if waiting_request is request_state:
                self.waiting.pop(index)
                return True
        return False

    def admit_waiting_requests(self) -> None:
        while self.waiting and self.has_free_slot():
            available_slots = self.max_slots - self.occupied_slot_count()

            selected_requests = self.scheduling_policy.select_admission(
                waiting=list(self.waiting),
                active=self.active_request_states(),
                available_slots=available_slots,
                kv_block_manager=self.kv_block_manager,
                decode_engine=self.decode_engine
            )

            if not selected_requests:
                return
            
            request_state = selected_requests[0]

            if not self.remove_from_waiting(request_state):
                raise RuntimeError(
                    f"Scheduling policy selected request_id={request_state.request_id!r} that is not currently waiting"
                )
            
            slot_index = self.first_free_slot_index()
            if slot_index is None:
                self.waiting.insert(0, request_state)
                return
            
            request_state.mark_admitted()

            try:
                prompt_tokens = self.decode_engine.count_prompt_tokens(
                    request_state.prompt
                )

                reserved_tokens = prompt_tokens + request_state.max_new_tokens
                if not self.kv_block_manager.can_allocate_tokens(reserved_tokens):
                    request_state.status = "waiting"
                    self.waiting.insert(0, request_state)
                    return
                
                request_id = str(request_state.request_id)
                block_table = self.kv_block_manager.allocate_for_tokens(request_id=request_id, num_tokens=prompt_tokens)

                request_state.block_table = block_table
                request_state.prompt_tokens = prompt_tokens

                self.decode_engine.init_request_state(request_state)
                request_state.num_computed_tokens = request_state.prompt_tokens
            
            except Exception as error:
                self.kv_block_manager.free(str(request_state.request_id))
                request_state.mark_failed(error)
                self.finished.append(request_state)
                self.metrics_store.record_finished(request_state)
                continue

            if self.occupied_slot_count() > 0:
                self.late_admissions += 1
            
            self.slots[slot_index] = request_state
            self.admitted_count += 1



    # -------------------------------------------------------------------------
    # Decode
    # -------------------------------------------------------------------------

    def decode_active_requests(self) -> None:
        active = self.active_request_states()
        if not active:
            return

        self.decode_steps += 1

        runnable: list[RequestState] = []
        stalled: list[RequestState] = []

        for request_state in active:
            request_id = str(request_state.request_id)
            token_position = request_state.prompt_tokens + request_state.generated_tokens

            try:
                self.kv_block_manager.ensure_capacity_for_token(
                    request_id=request_id,
                    token_position=token_position,
                )
                request_state.block_table = self.kv_block_manager.get_block_tables(
                    request_id
                )
                runnable.append(request_state)

            except KVBlockAllocationError:
                self.decode_stalls += 1
                self.kv_allocation_failures += 1
                stalled.append(request_state)

        if not runnable:
            if stalled:
                self.evict_one_stalled_request(stalled)
            return
        
        decode_batch = self.scheduling_policy.select_decode_batch(
            active=runnable,
            kv_block_manager=self.kv_block_manager,
            max_batch_size=None
        )

        if not decode_batch:
            self.decode_stalls += 1

        try:
            output = self.decode_engine.decode_step(
                request_states=decode_batch,
                kv_block_manager=self.kv_block_manager,
            )
            self.apply_decode_output(output)

        except Exception as error:
            for request_state in decode_batch:
                self.fail_request(request_state, error)

    def apply_decode_output(self, output: DecodeStepOutput) -> None:
        self.last_backend_ms = output.backend_ms
        self.last_decode_batch_snapshot = output.decode_batch_snapshot

        if output.decode_batch_snapshot is not None:
            self.decode_batches_built += 1

        request_by_id = {
            str(request_state.request_id): request_state
            for request_state in self.active_request_states()
        }

        for request_output in output.request_outputs:
            request_state = request_by_id.get(str(request_output.request_id))
            if request_state is None:
                continue

            if request_output.text:
                request_state.append_text(request_output.text)
            else:
                request_state.mark_first_token()

            self.tokens_generated += request_output.generated_tokens
            request_state.num_computed_tokens = (
                request_state.prompt_tokens + request_state.generated_tokens
            )

            if request_output.finished:
                self.finish_request(request_state)

    # -------------------------------------------------------------------------
    # Finish / fail / evict
    # -------------------------------------------------------------------------

    def finish_request(self, request_state: RequestState) -> None:
        request_id = str(request_state.request_id)

        self.kv_block_manager.free(request_id)
        request_state.mark_finished()

        self.finished.append(request_state)
        self.metrics_store.record_finished(request_state)

        for index, slot_request in enumerate(self.slots):
            if slot_request is request_state:
                self.slots[index] = None
                break

        self.early_finishes += 1

    def fail_request(self, request_state: RequestState, error: Exception) -> None:
        self.kv_block_manager.free(str(request_state.request_id))
        request_state.mark_failed(error)

        self.finished.append(request_state)
        self.metrics_store.record_finished(request_state)

        for index, slot_request in enumerate(self.slots):
            if slot_request is request_state:
                self.slots[index] = None
                break

    def evict_one_stalled_request(self, stalled: list[RequestState]) -> None:
        victim = stalled[-1]

        self.kv_block_manager.free(str(victim.request_id))
        victim.mark_failed(
            RuntimeError(
                "KV cache exhausted: request evicted to break decode-time "
                "memory deadlock"
            )
        )

        self.finished.append(victim)
        self.metrics_store.record_finished(victim)

        for index, slot_request in enumerate(self.slots):
            if slot_request is victim:
                self.slots[index] = None
                break

        self.kv_oom_evictions += 1

    # -------------------------------------------------------------------------
    # Step loop
    # -------------------------------------------------------------------------

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