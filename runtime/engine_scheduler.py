import time
from queue import Empty
from threading import Lock, Thread
from typing import Optional

from .metrics_store import MetricsStore
from .decode_engine import DecodeEngine, DecodeStepOutput
from .kv_block_manager import KVBlockAllocationError, KVBlockManager
from .request_queue import RequestQueue
from .request_state import RequestState
from .scheduling_policy import FCFSPolicy, SchedulingPolicy
from .scheduler_work_plan import (
    build_decode_work_plan,
    build_prefill_work_plan,
    requests_from_work_plan,
    summarize_work_plan,
)


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
        - Custom engines
    """

    def __init__(
        self,
        decode_engine: DecodeEngine,
        request_queue: RequestQueue,
        metrics_store: MetricsStore,
        kv_block_manager: KVBlockManager,
        max_slots: int = 4,
        scheduling_policy: SchedulingPolicy | None = None,
        max_scheduled_tokens_per_step: int | None = None,
        step_sleep_seconds: float = 0.0,
        idle_sleep_seconds: float = 0.01,
    ) -> None:
        self.decode_engine = decode_engine
        self.request_queue = request_queue
        self.metrics_store = metrics_store
        self.kv_block_manager = kv_block_manager
        self.scheduling_policy = scheduling_policy or FCFSPolicy()

        self.max_slots = max_slots
        self.max_scheduled_tokens_per_step = max_scheduled_tokens_per_step
        self.step_sleep_seconds = step_sleep_seconds
        self.idle_sleep_seconds = idle_sleep_seconds

        self.waiting: list[RequestState] = []
        self.finished: list[RequestState] = []
        self.slots: list[Optional[RequestState]] = [
            None for _ in range(max_slots)
        ]

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

        self.last_candidate_work_plan_summary: dict | None = None
        self.last_executed_work_plan_summary: dict | None = None

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
        return sum(
            1
            for request_state in self.slots
            if request_state is not None
        )

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
            active_requests = self.active_request_states()

            active_prefill_tokens_remaining = sum(
                request.prefill_tokens_remaining
                for request in active_requests
            )
            active_decode_tokens_remaining = sum(
                request.decode_tokens_remaining
                for request in active_requests
            )
            active_estimated_tokens_remaining = sum(
                request.estimated_total_tokens_remaining
                for request in active_requests
            )

            waiting_prefill_tokens_remaining = sum(
                request.prefill_tokens_remaining
                for request in self.waiting
            )
            waiting_decode_tokens_remaining = sum(
                request.decode_tokens_remaining
                for request in self.waiting
            )
            waiting_estimated_tokens_remaining = sum(
                request.estimated_total_tokens_remaining
                for request in self.waiting
            )

            return {
                "engine_step": self.engine_step,
                "scheduler_type": "engine_agnostic_continuous_slots",
                "policy_name": self.scheduling_policy.name,
                "engine": type(self.decode_engine).__name__,
                "waiting": len(self.waiting),
                "active": self.occupied_slot_count(),
                "finished": len(self.finished),
                "max_slots": self.max_slots,
                "max_scheduled_tokens_per_step": (
                    self.max_scheduled_tokens_per_step
                ),
                "active_prefill_tokens_remaining": (
                    active_prefill_tokens_remaining
                ),
                "active_decode_tokens_remaining": (
                    active_decode_tokens_remaining
                ),
                "active_estimated_tokens_remaining": (
                    active_estimated_tokens_remaining
                ),
                "waiting_prefill_tokens_remaining": (
                    waiting_prefill_tokens_remaining
                ),
                "waiting_decode_tokens_remaining": (
                    waiting_decode_tokens_remaining
                ),
                "waiting_estimated_tokens_remaining": (
                    waiting_estimated_tokens_remaining
                ),
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
                        "num_computed_tokens": (
                            request_state.num_computed_tokens
                        ),
                        "prefill_tokens_total": (
                            request_state.prefill_tokens_total
                        ),
                        "prefill_tokens_remaining": (
                            request_state.prefill_tokens_remaining
                        ),
                        "decode_tokens_total": (
                            request_state.decode_tokens_total
                        ),
                        "decode_tokens_remaining": (
                            request_state.decode_tokens_remaining
                        ),
                        "estimated_total_tokens_remaining": (
                            request_state.estimated_total_tokens_remaining
                        ),
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
                "last_candidate_work_plan_summary": (
                    self.last_candidate_work_plan_summary
                ),
                "last_executed_work_plan_summary": (
                    self.last_executed_work_plan_summary
                ),
                "late_admissions": self.late_admissions,
                "early_finishes": self.early_finishes,
                "queue_length_history_tail": self.queue_length_history[-20:],
                "occupied_slots_history_tail": (
                    self.occupied_slots_history[-20:]
                ),
                "finished_count_history_tail": (
                    self.finished_count_history[-20:]
                ),
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

    def reserve_kv_for_request(self, request_state: RequestState) -> bool:
        """
        Reserve initial KV blocks for an admitted request.

        This method performs scheduler-owned memory admission logic:
            - count prompt tokens
            - check whether prompt + decode reservation can fit
            - allocate prompt KV blocks
            - store prompt token count and block table on RequestState

        The actual prefill computation is owned by the DecodeEngine.
        """

        prompt_tokens = self.decode_engine.count_prompt_tokens(
            request_state.prompt
        )
        reserved_tokens = prompt_tokens + request_state.max_new_tokens

        if not self.kv_block_manager.can_allocate_tokens(reserved_tokens):
            return False

        request_id = str(request_state.request_id)
        block_table = self.kv_block_manager.allocate_for_tokens(
            request_id=request_id,
            num_tokens=prompt_tokens,
        )

        request_state.block_table = block_table
        request_state.prompt_tokens = prompt_tokens
        return True

    def prefill_admitted_request(self, request_state: RequestState) -> None:
        """
        Run full prefill for an admitted request.

        The scheduler decides when this work happens. The DecodeEngine owns the
        model-specific execution and KV-cache materialization.

        Today this is a full prompt prefill. Later, this boundary is where
        chunked prefill scheduling can be introduced.
        """

        self.decode_engine.prefill_request(request_state)

    def mark_request_prefill_complete(self, request_state: RequestState) -> None:
        """
        Mark a request as ready for decode after prefill completes.

        Today full prefill completes immediately during admission. Later, chunked
        prefill will call this only when num_computed_tokens reaches prompt_tokens.
        """
        if request_state.prefill_tokens_remaining != 0:
            raise RuntimeError(
                f"Cannot mark request_id={request_state.request_id!r}"
                "decode-ready while prefill tokens remain: "
                f"{request_state.prefill_tokens_remaining}"
            )
        
        request_state.mark_decoding()

    def place_request_in_slot(
        self,
        *,
        request_state: RequestState,
        slot_index: int,
    ) -> None:
        """
        Activate an admitted and prefilled request in a scheduler slot.
        """

        self.slots[slot_index] = request_state
        self.admitted_count += 1

    def remove_from_waiting(self, request_state: RequestState) -> bool:
        for index, waiting_request in enumerate(self.waiting):
            if waiting_request is request_state:
                self.waiting.pop(index)
                return True
        return False

    def admit_waiting_requests(self) -> None:
        while self.waiting and self.has_free_slot():
            available_slots = self.max_slots - self.occupied_slot_count()

            selected_requests = self.scheduling_policy.select_admissions(
                waiting=list(self.waiting),
                active=self.active_request_states(),
                available_slots=available_slots,
                kv_block_manager=self.kv_block_manager,
                decode_engine=self.decode_engine,
            )

            if not selected_requests:
                return

            admitted_any = False

            for request_state in selected_requests:
                if not self.has_free_slot():
                    return

                if not self.remove_from_waiting(request_state):
                    raise RuntimeError(
                        "Scheduling policy selected "
                        f"request_id={request_state.request_id!r} "
                        "that is not currently waiting"
                    )

                slot_index = self.first_free_slot_index()
                if slot_index is None:
                    self.waiting.insert(0, request_state)
                    return

                request_state.mark_admitted()

                try:
                    if not self.reserve_kv_for_request(request_state):
                        request_state.status = "waiting"
                        self.waiting.insert(0, request_state)
                        continue

                except Exception as error:
                    self.kv_block_manager.free(str(request_state.request_id))
                    request_state.mark_failed(error)
                    self.finished.append(request_state)
                    self.metrics_store.record_finished(request_state)
                    continue

                if self.occupied_slot_count() > 0:
                    self.late_admissions += 1

                self.place_request_in_slot(
                    request_state=request_state,
                    slot_index=slot_index,
                )
                admitted_any = True

            if not admitted_any:
                return

    # -------------------------------------------------------------------------
    # Decode
    # -------------------------------------------------------------------------

    def build_runnable_decode_requests(
        self,
        active: list[RequestState],
    ) -> tuple[list[RequestState], list[RequestState]]:
        """
        Split active requests into runnable and stalled decode candidates.

        A request is runnable if the KV block manager can provide capacity for
        the next generated token position. Stalled requests could not allocate
        decode-time KV capacity this step.
        """

        runnable: list[RequestState] = []
        stalled: list[RequestState] = []

        for request_state in active:
            request_id = str(request_state.request_id)
            token_position = (
                request_state.prompt_tokens + request_state.generated_tokens
            )

            try:
                self.kv_block_manager.ensure_capacity_for_token(
                    request_id=request_id,
                    token_position=token_position,
                )
                request_state.block_table = self.kv_block_manager.get_block_tables(
                    request_id=request_id
                )
                runnable.append(request_state)

            except KVBlockAllocationError:
                self.decode_stalls += 1
                self.kv_allocation_failures += 1
                stalled.append(request_state)

        return runnable, stalled
    
    def decode_ready_request_states(self) -> list[RequestState]:
        """
        Return active requests that are ready for decode.

        A request is decode-ready once full prefill has completed and the request is 
        in decoding state.
        """
        return [
            request_state
            for request_state in self.active_request_states()
            if(
                request_state.status == "decoding"
                and request_state.prefill_tokens_remaining == 0
            )
        ]
    
    def prefill_active_request_states(self) -> list[RequestState]:
        """
        Return active requests that still need prefill work

        Today this is usually empty because admitted reqeusts are fully prefetched immediately. 
        Later, chunked prefill will keep requests in this state across scheduler steps.
        """
        return [
            request_state
            for request_state in self.active_request_states()
            if(
                request_state.status == "prefill"
                and request_state.prefill_tokens_remaining > 0
            )
        ]
    
    def run_prefill_for_active_requests(self) -> None:
        """
        Run full prefill for active requests that are still in prefill state.

        Prefill is budgeted by scheduler work items. A request remains in prefill
        state until all prompt tokens have been computed.
        """

        prefill_requests = self.prefill_active_request_states()
        if not prefill_requests:
            return
        
        work_plan = build_prefill_work_plan(prefill_requests, max_scheduled_tokens=self.max_scheduled_tokens_per_step)
        self.last_candidate_work_plan_summary = summarize_work_plan(work_plan)
        
        for work in work_plan:
            request_state = work.request_state

            try:
                self.decode_engine.prefill_chunk(
                    request_state=request_state,
                    num_tokens=work.num_scheduled_tokens,
                    kv_block_manager=self.kv_block_manager
                )

                if request_state.prefill_tokens_remaining == 0:
                    self.mark_request_prefill_complete(request_state)
            
            except Exception as error:
                self.fail_request(request_state, error)
        
        self.last_executed_work_plan_summary = summarize_work_plan(work_plan)

    def select_decode_batch(
        self,
        runnable: list[RequestState],
    ) -> list[RequestState]:
        """
        Select the runnable requests that should execute this decode step.

        The scheduler delegates ordering/budgeting decisions to the configured
        SchedulingPolicy.
        """

        return self.scheduling_policy.select_decode_batch(
            active=runnable,
            kv_block_manager=self.kv_block_manager,
            max_batch_size=None,
        )

    def run_decode_batch(
        self,
        decode_batch: list[RequestState],
    ) -> DecodeStepOutput:
        """
        Execute one decode batch through the DecodeEngine.

        The scheduler owns batch selection. The DecodeEngine owns model
        execution.
        """

        return self.decode_engine.decode_step(
            request_states=decode_batch,
            kv_block_manager=self.kv_block_manager,
        )

    def decode_active_requests(self) -> None:
        active = self.decode_ready_request_states()
        if not active:
            return

        self.decode_steps += 1

        runnable, stalled = self.build_runnable_decode_requests(active=active)

        if not runnable:
            if stalled:
                self.evict_one_stalled_request(stalled)
            return

        decode_batch = self.select_decode_batch(runnable)

        if not decode_batch:
            self.decode_stalls += 1
            return

        work_plan = build_decode_work_plan(decode_batch)
        self.last_candidate_work_plan_summary = summarize_work_plan(work_plan)

        decode_batch = requests_from_work_plan(work_plan)
        self.last_executed_work_plan_summary = summarize_work_plan(work_plan)

        try:
            output = self.run_decode_batch(decode_batch)
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

    def fail_request(
        self,
        request_state: RequestState,
        error: Exception,
    ) -> None:
        self.kv_block_manager.free(str(request_state.request_id))
        request_state.mark_failed(error)

        self.finished.append(request_state)
        self.metrics_store.record_finished(request_state)

        for index, slot_request in enumerate(self.slots):
            if slot_request is request_state:
                self.slots[index] = None
                break

    def evict_one_stalled_request(
        self,
        stalled: list[RequestState],
    ) -> None:
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
            self.run_prefill_for_active_requests()
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