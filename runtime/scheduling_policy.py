from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from runtime.decode_engine import DecodeEngine
from runtime.kv_block_manager import KVBlockManager
from runtime.request_state import RequestState


class SchedulingPolicy(Protocol):
    name: str

    def select_admission(
            self,
            *,
            waiting: list[RequestState],
            active: list[RequestState],
            available_slots: int,
            kv_block_manager: KVBlockManager,
            decode_engine: DecodeEngine
    ) -> list[RequestState]:
        ...
    

    def select_decode_batch(
            self,
            *,
            active: list[RequestState],
            kv_block_manager: KVBlockManager,
            max_batch_size: int | None = None
    ) -> list[RequestState]:
        ...


@dataclass(frozen=True)
class FCFSPolicy:
    name: str = "fcfs"

    def select_admission(
            self,
            *,
            waiting: list[RequestState],
            active: list[RequestState],
            available_slots: int,
            kv_block_manager: KVBlockManager,
            decode_engine: DecodeEngine
    ) -> list[RequestState]:
        if available_slots <= 0:
            return []
        
        return waiting[:available_slots]
    

    def select_decode_batch(
            self,
            *,
            active: list[RequestState],
            kv_block_manager: KVBlockManager,
            max_batch_size: int | None = None
    ) -> list[RequestState]:
        
        if max_batch_size is None:
            return list(active)
        
        return list(active[:max_batch_size])
    

@dataclass(frozen=True)
class DecodeBudgetPolicy:
    max_decode_batch_size: int

    @property
    def name(self) -> str:
        return f"decode_budget_{self.max_decode_batch_size}"

    def select_admission(
        self,
        *,
        waiting: list[RequestState],
        active: list[RequestState],
        available_slots: int,
        kv_block_manager: KVBlockManager,
        decode_engine: DecodeEngine,
    ) -> list[RequestState]:
        if available_slots <= 0:
            return []

        return list(waiting[:available_slots])

    def select_decode_batch(
        self,
        *,
        active: list[RequestState],
        kv_block_manager: KVBlockManager,
        max_batch_size: int | None = None,
    ) -> list[RequestState]:
        if not active:
            return []

        if self.max_decode_batch_size <= 0:
            return []

        effective_limit = self.max_decode_batch_size

        if max_batch_size is not None:
            effective_limit = min(effective_limit, max_batch_size)

        return list(active[:effective_limit])
        

    
