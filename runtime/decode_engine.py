from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .kv_block_manager import KVBlockManager
from .request_state import RequestState


@dataclass(frozen=True)
class RequestDecodeOutput:
    request_id: str
    text: str
    generated_tokens: int
    finished: bool


@dataclass(frozen=True)
class DecodeStepOutput:
    request_outputs: list[RequestDecodeOutput]
    backend_ms: float | None = None
    decode_batch_snapshot: dict | None = None


class DecodeEngine(Protocol):
    """
    Execution boundary used by the scheduler.

    The scheduler owns:
        - request admission
        - active slots
        - queueing
        - K/V block reservation/free policy
        - finish/free policy
        - metrics

    The DecodeEngine owns:
        - tokenization and prompt length estimation
        - request-local model state initialization
        - prefill execution
        - one decode step of execution
        - backend-specific model/KV behavior
    
    """

    @property
    def device(self) -> str:
        ...
    
    def count_prompt_tokens(self, prompt: str) -> int:
        ...
    
    def init_request_state(self, request_state: RequestState) -> None:
        ...

    def prefill_request(self, request_state: RequestState) -> None:
        """
        Run full prefill for one admitted request

        Current implementation may perform tokenization, prompt forward pass,
        KV-cache materialization, and first next-token selection in this method

        Later, chunked prefill can replace this full-prefill operation with
        incremental prefill progress while preserving the scheduler/engine
        ownership boundary
        """
        ...

    def prefill_chunk(
            self,
            request_state: RequestState,
            num_tokens: int,
            kv_block_manager,
    ) -> None:
        ...
    
    def decode_step(
            self,
            request_states: list[RequestState],
            kv_block_manager: KVBlockManager,
    ) -> DecodeStepOutput:
        ...