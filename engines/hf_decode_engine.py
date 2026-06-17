from __future__ import annotations

import time

from runtime.decode_engine import DecodeStepOutput, RequestDecodeOutput
from runtime.kv_block_manager import KVBlockManager
from engines.model_runner import ModelRunner
from runtime.request_state import RequestState


class HFDecodeEngine:
    """
    DecodeEngine implementation backed by the existing Hugging Face ModelRunner.

    This preserves the current real-text-generation path while hiding Hugging Face
    behind the DecodeEngine boundary.

    Important:
        ModelRunner.decode_one_token currently mutates:
            - request_state.past_key_values
            - request_state.next_token
            - request_state.generated_tokens

        So this engine treats token accounting as engine-owned.
        The scheduler should not increment generated_tokens again.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self.runner = runner

    @property
    def device(self) -> str:
        return self.runner.device

    def count_prompt_tokens(self, prompt: str) -> int:
        return self.runner.count_prompt_tokens(prompt)

    def init_request_state(self, request_state: RequestState) -> None:
        self.runner.init_request_state(request_state)

    def decode_step(
        self,
        request_states: list[RequestState],
        kv_block_manager: KVBlockManager,
    ) -> DecodeStepOutput:
        del kv_block_manager  # HF owns real KV internally through past_key_values.

        t0 = time.perf_counter()
        outputs: list[RequestDecodeOutput] = []

        for request_state in request_states:
            if request_state.status == "finished":
                continue

            before_generated = request_state.generated_tokens

            text = self.runner.decode_one_token(request_state)

            after_generated = request_state.generated_tokens
            generated_delta = after_generated - before_generated

            if generated_delta != 1:
                raise RuntimeError(
                    "HFDecodeEngine expected exactly one generated token. "
                    f"request_id={request_state.request_id}, "
                    f"before={before_generated}, after={after_generated}, "
                    f"delta={generated_delta}"
                )

            outputs.append(
                RequestDecodeOutput(
                    request_id=str(request_state.request_id),
                    text=text,
                    generated_tokens=generated_delta,
                    finished=request_state.is_finished(),
                )
            )

        t1 = time.perf_counter()

        return DecodeStepOutput(
            request_outputs=outputs,
            backend_ms=(t1 - t0) * 1000.0,
            decode_batch_snapshot=None,
        )