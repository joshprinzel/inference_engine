import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BatchMetrics:
    batch_size: int
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    created_time: float = field(default_factory=time.perf_counter)
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None

    decode_time_seconds_total: float = 0.0
    decode_steps: int = 0
    tokens_generated: int = 0

    finished_time: Optional[float] = None
    status: str = "running"

    def mark_prefill_start(self) -> None:
        self.prefill_start_time = time.perf_counter()
    
    def mark_prefill_end(self) -> None:
        self.prefill_end_time = time.perf_counter()

    def record_decode_step(
            self,
            decode_time_seconds: float,
            tokens_generated_this_step: int,
    ) -> None:
        self.decode_time_seconds_total += decode_time_seconds
        self.decode_steps += 1
        self.tokens_generated += tokens_generated_this_step
    
    def mark_finished(self) -> None:
        self.finished_time = time.perf_counter()
        self.status = "finished"
    
    def mark_failed(self) -> None:
        self.finished_time = time.perf_counter()
        self.status = "failed"
    
    @property
    def prefill_time_seconds(self) -> float:
        if self.prefill_start_time is None or self.prefill_end_time is None:
            return 0.0

        return self.prefill_end_time - self.prefill_start_time

    @property
    def total_time_seconds(self) -> float:
        if self.finished_time is None:
            return 0.0
        return self.finished_time - self.created_time
    
    @property
    def batch_tokens_per_second(self) -> float:
        if self.decode_time_seconds_total <= 0:
            return 0.0
        return self.tokens_generated / self.decode_time_seconds_total
    
    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "batch_size": self.batch_size,
            "prefill_time_seconds": self.prefill_time_seconds,
            "decode_time_seconds_total": self.decode_time_seconds_total,
            "decode_steps": self.decode_steps,
            "tokens_generated": self.tokens_generated,
            "batch_tokens_per_second": self.batch_tokens_per_second,
            "total_time_seconds": self.total_time_seconds,
        }