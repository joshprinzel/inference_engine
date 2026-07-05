import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Optional, Any

@dataclass
class RequestState:
    prompt: str
    max_new_tokens: int

    request_id: str = field(default_factory=lambda: str(uuid.uuid4())) #UUID = Universally Unique Identifier

    arrival_time: float = field(default_factory=time.perf_counter)
    admit_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    status: str = "waiting"

    generated_text_parts: list[str] = field(default_factory=list)
    generated_tokens: int = 0
    output_queue: Queue = field(default_factory=Queue)

    error: Optional[str] = None

    # Real Decode State owned by the engine
    input_ids: Any = None
    attention_mask: Any = None #Attention mask pretty much makes the model learn step by step without looking ahead to future tokens
    past_key_values: Any = None
    next_token: Any = None
    prompt_tokens: int = 0

    block_table: list[int] | None = None
    num_computed_tokens: int = 0

    def mark_admitted(self) -> None:
        if self.admit_time is None:
            self.admit_time = time.perf_counter()
        self.status = "prefill"
    
    def mark_decoding(self) -> None:
        self.status = "decoding"

    def mark_first_token(self) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def append_text(self, text: str) -> None:
        self.mark_first_token()
        self.generated_text_parts.append(text)
        self.output_queue.put(text)

    def mark_finished(self) -> None:
        if self.finish_time is None:
            self.finish_time = time.perf_counter()
        self.status = "finished"
        self.output_queue.put(None)
    
    def mark_stream_finished(self) -> None:
        if self.finish_time is None:
            self.finish_time = time.perf_counter()
        self.status = "finished"
        self.output_queue.put(None)
    
    def mark_failed(self, error: Exception) -> None:
        if self.finish_time is None:
            self.finish_time = time.perf_counter()
        self.status = "failed"
        self.error = repr(error)
        self.output_queue.put(None)

    def is_finished(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens
    

    @property
    def generated_text(self) -> str:
        return "".join(self.generated_text_parts)
    
    @staticmethod
    def _seconds_to_ms(value: float | None) -> float | None:
        if value is None:
            return None
        return value * 1000.0
    
    @property
    def queue_wait_seconds(self) -> float | None:
        if self.admit_time is None:
            return None
        return self.admit_time - self.arrival_time

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def decode_latency_seconds(self) -> float | None:
        if self.first_token_time is None or self.finish_time is None:
            return None
        return self.finish_time - self.first_token_time

    @property
    def latency_seconds(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @staticmethod
    def _seconds_to_ms(value: float | None) -> float | None:
        if value is None:
            return None
        return value * 1000.0

    @property
    def queue_wait_ms(self) -> float | None:
        return self._seconds_to_ms(self.queue_wait_seconds)

    @property
    def ttft_ms(self) -> float | None:
        return self._seconds_to_ms(self.ttft_seconds)

    @property
    def decode_latency_ms(self) -> float | None:
        return self._seconds_to_ms(self.decode_latency_seconds)

    @property
    def latency_ms(self) -> float | None:
        return self._seconds_to_ms(self.latency_seconds)
