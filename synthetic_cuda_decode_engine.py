from __future__ import annotations

import time

import torch

from attention_backend import AttentionBackend, build_attention_backend
from decode_batch import build_decode_batch
from decode_engine import DecodeStepOutput, RequestDecodeOutput
from kv_block_manager import KVBlockManager
from kv_cache_layout import KVCacheLayout
from kv_cache_pool import KVCachePool
from request_state import RequestState


class SyntheticCudaDecodeEngine:
    """
    DecodeEngine implementation that exercises the custom CUDA paged attention path.

    This is not full model execution.

    Synthetic:
        - q tensors
        - generated token ids
        - generated-token K/V writes

    Real:
        - RequestState lifecycle
        - KVBlockManager block tables
        - DecodeBatch lowering
        - KVCachePool physical layout
        - AttentionBackend
        - CUDA paged attention kernel
    """

    def __init__(
        self,
        total_blocks: int,
        block_size_tokens: int,
        num_layers: int = 1,
        num_query_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        dtype: str = "float16",
        device: str = "cuda",
        attention_backend: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

        self._device = device
        self.layer_id = 0

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.layout = KVCacheLayout(
            num_layers=num_layers,
            total_blocks=total_blocks,
            block_size_tokens=block_size_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        self.cache_pool = KVCachePool(self.layout)
        self.cache_pool.zero_()

        self.attention_backend: AttentionBackend = build_attention_backend(
            attention_backend
        )

    @property
    def device(self) -> str:
        return self._device

    def count_prompt_tokens(self, prompt: str) -> int:
        # Synthetic engine does not tokenize real text.
        # Keep this deterministic and simple.
        del prompt
        return 128

    def init_request_state(self, request_state: RequestState) -> None:
        if request_state.block_table is None:
            raise ValueError(
                f"request_id={request_state.request_id} has no block_table"
            )

        self._fill_prompt_kv(
            block_table=request_state.block_table,
            prompt_tokens=request_state.prompt_tokens,
        )

        request_state.next_token = torch.tensor(
            [[100]],
            dtype=torch.int64,
        )
        request_state.num_computed_tokens = request_state.prompt_tokens
        request_state.mark_decoding()

    def decode_step(
        self,
        request_states: list[RequestState],
        kv_block_manager: KVBlockManager,
    ) -> DecodeStepOutput:
        if not request_states:
            return DecodeStepOutput(request_outputs=[])

        decode_batch = build_decode_batch(
            request_states=request_states,
            kv_block_manager=kv_block_manager,
            device=self.device,
        )

        q = torch.randn(
            decode_batch.batch_size,
            self.num_query_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )

        if self.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = self.attention_backend.decode(
                q=q,
                cache_pool=self.cache_pool,
                layer_id=self.layer_id,
                block_tables=decode_batch.block_tables,
                seq_lens=decode_batch.seq_lens,
            )
            end.record()

            torch.cuda.synchronize()
            backend_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            _ = self.attention_backend.decode(
                q=q,
                cache_pool=self.cache_pool,
                layer_id=self.layer_id,
                block_tables=decode_batch.block_tables,
                seq_lens=decode_batch.seq_lens,
            )
            t1 = time.perf_counter()
            backend_ms = (t1 - t0) * 1000.0

        outputs: list[RequestDecodeOutput] = []

        for request_state in request_states:
            if request_state.block_table is None:
                raise ValueError(
                    f"request_id={request_state.request_id} has no block_table"
                )

            token_position = (
                request_state.prompt_tokens + request_state.generated_tokens
            )

            # EngineScheduler already ensured capacity before decode_step.
            # Refresh the block table in case a new block was appended.
            block_table = kv_block_manager.get_block_tables(
                str(request_state.request_id)
            )
            request_state.block_table = block_table

            self._write_generated_token_kv(
                block_table=block_table,
                token_position=token_position,
            )

            request_state.generated_tokens += 1
            request_state.num_computed_tokens = (
                request_state.prompt_tokens + request_state.generated_tokens
            )

            request_state.next_token = torch.tensor(
                [[1000 + request_state.generated_tokens]],
                dtype=torch.int64,
            )

            finished = request_state.is_finished()

            outputs.append(
                RequestDecodeOutput(
                    request_id=str(request_state.request_id),
                    text=f"<tok{request_state.generated_tokens}>",
                    generated_tokens=1,
                    finished=finished,
                )
            )

        return DecodeStepOutput(
            request_outputs=outputs,
            backend_ms=backend_ms,
            decode_batch_snapshot=decode_batch.snapshot(),
        )

    def _fill_prompt_kv(
        self,
        block_table: list[int],
        prompt_tokens: int,
    ) -> None:
        for token_position in range(prompt_tokens):
            key = torch.randn(
                self.num_kv_heads,
                self.head_dim,
                device=self.device,
                dtype=self.cache_pool.key_cache.dtype,
            )
            value = torch.randn_like(key)

            self.cache_pool.write_request_token(
                layer_id=self.layer_id,
                block_table=block_table,
                token_position=token_position,
                key=key,
                value=value,
            )

    def _write_generated_token_kv(
        self,
        block_table: list[int],
        token_position: int,
    ) -> None:
        key = torch.randn(
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=self.cache_pool.key_cache.dtype,
        )
        value = torch.randn_like(key)

        self.cache_pool.write_request_token(
            layer_id=self.layer_id,
            block_table=block_table,
            token_position=token_position,
            key=key,
            value=value,
        )