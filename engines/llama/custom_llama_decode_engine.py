from __future__ import annotations


import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from runtime.decode_engine import DecodeStepOutput, RequestDecodeOutput
from runtime.kv_block_manager import KVBlockManager
from runtime.request_state import RequestState

from runtime.kv_cache_layout import KVCacheLayout
from runtime.kv_cache_pool import KVCachePool
from runtime.attention_backend import AttentionBackend, build_attention_backend
from runtime.kv_cache_transfer import write_past_key_values_to_pool
from engines.llama.paged_model import llama_model_decode_batch_with_paged_attention_from_hf_weights

from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"




class CustomLlamaDecodeEngine:
    """
    Custom Llama DecodeEngine.
    """

    def __init__(
            self,
            model_id: str = DEFAULT_MODEL_ID,
            device: str | None = None,
            dtype: torch.dtype | None = None,
            total_kv_blocks: int = 256,
            block_size_tokens: int = 16,
            attention_backend_name: str = "cuda",
            ) -> None:
        self.model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self._device == "cuda" else torch.float32)

        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self._device)

        self.model.eval()
        
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        dtype_name = self._dtype_name(self.dtype)
        self.kv_cache_pool = KVCachePool(
            KVCacheLayout(
                num_layers=self.config.num_hidden_layers,
                total_blocks=total_kv_blocks,
                block_size_tokens=block_size_tokens,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                dtype=dtype_name,
                device=self._device,
            )
        )
        self.kv_cache_pool.zero_()

        self.attention_backend_name = attention_backend_name
        self.attention_backend: AttentionBackend = build_attention_backend(attention_backend_name)

    @property
    def device(self) -> str:
        return self._device
    
    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        if dtype == torch.float16:
            return "float16"
        if dtype == torch.bfloat16:
            return "bfloat16"
        if dtype == torch.float32:
            return "float32"

        raise ValueError(f"Unsupported dtype: {dtype}")
    
    def count_prompt_tokens(self, prompt: str) -> int:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return int(input_ids.shape[-1])
    
    def _build_block_tables_tensor(
            self,
            block_tables: list[list[int]]
    ) -> torch.Tensor:
        max_blocks = max(len(block_table) for block_table in block_tables)
        if max_blocks == 0:
            raise ValueError("Cannot build block_tables tensor with no blocks")
        
        rows = [
            block_table + [-1] * (max_blocks - len(block_table))
            for block_table in block_tables
        ]
        return torch.tensor(
            rows,
            device=self._device,
            dtype=torch.int32
        )
    
    def _ensure_prompt_input_ids(
        self,
        request_state: RequestState,
    ) -> torch.Tensor:
        """
        Ensure request_state.input_ids contains the tokenized prompt.

        During chunked prefill, this lets us tokenize once and reuse slices across
        scheduler steps.
        """

        input_ids = getattr(request_state, "input_ids", None)

        if input_ids is None:
            encoded = self.tokenizer(
                request_state.prompt,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]

        input_ids = input_ids.to(self._device)

        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Expected request_state.input_ids to have shape [1, seq_len], "
                f"got {tuple(input_ids.shape)}"
            )

        request_state.input_ids = input_ids
        request_state.prompt_tokens = int(input_ids.shape[-1])
        return input_ids
    
    def _validate_prefill_request_state(
            self,
            request_state: RequestState
    ) -> None:
        """
        Validate scheduler-owned state required before materializing prompt KV.
        """
        if request_state.block_table is None:
            raise ValueError(
                f"request_state.block_table is None for "
                f"request_id={request_state.request_id!r}. "
                "Scheduler must allocate KV blocks before prefill."
            )
        
    def _run_full_prefill_forward(
            self,
            input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        """
        Run the current full-prompt prefill forward path.
        """

        logits, past_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
            hf_model=self.model,
            input_ids=input_ids,
            past_key_values=None
        )
        return logits, past_key_values
    
    def _write_prefill_kv_to_pool(
            self,
            *,
            request_state: RequestState,
            past_key_values,
            start_token_position: int
    ) -> None:
        """
        Write prefill K/V tensors into the physical KV cache pool.
        """

        if request_state.block_table is None:
            raise ValueError(
                f"request_state.block_table is None for "
                f"request_id={request_state.request_id!r}. "
                "Scheduler must allocate KV blocks before KV write."
            )
        
        write_past_key_values_to_pool(
            kv_cache_pool=self.kv_cache_pool,
            block_table=request_state.block_table,
            past_key_values=past_key_values,
            start_token_position=start_token_position
        )

    def _select_next_token_from_logits(
            self,
            logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Select the next decode token from the final position logits.
        """
        return torch.argmax(
            logits[:,-1,:],
            dim=-1,
            keepdim=True
        )
    
    def _slice_new_past_key_values(
            self,
            *,
            past_key_values,
            start_token_position: int,
            end_token_position: int
    ):
        """
        Extract only the newly computed K/V range from append-style present K/V.

        llama_model_forward_with_kv_cache_from_hf_weights return present K/V with
        shape [batch, kv_heads, past_len + q_len, head_dim]. For chunked prefill,
        we only want to write the newly computed suffix into the physical KV pool
        """
        return [
            (
                key[:, :, start_token_position:end_token_position, :].contiguous(),
                value[:, :, start_token_position:end_token_position, :].contiguous()
            )
            for key, value in past_key_values
        ]
    
    def prefill_request(self, request_state: RequestState) -> None:
        """
        Run full prompt prefill for one request and materialize KV into the pool.

        This remains a full-prefill compatibility path. Chunked prefill should use
        prefill_chunk(...).
        """

        self._validate_prefill_request_state(request_state)
        input_ids = self._ensure_prompt_input_ids(request_state)

        with torch.inference_mode():
            logits, past_key_values = self._run_full_prefill_forward(input_ids)
            next_token = self._select_next_token_from_logits(logits)
        
        self._write_prefill_kv_to_pool(
            request_state=request_state,
            past_key_values=past_key_values,
            start_token_position=0
        )

        request_state.past_key_values = None
        request_state.next_token = next_token
        request_state.generated_tokens = 0
        request_state.num_computed_tokens = request_state.prompt_tokens
    
    def prefill_chunk(
        self,
        request_state: RequestState,
        num_tokens: int,
        kv_block_manager: KVBlockManager,
    ) -> None:
        """
        Run one scheduled prefill chunk.

        This supports real multi-chunk prompt prefill:
            - tokenize prompt once
            - run only the scheduled prompt slice
            - use request-local contiguous past K/V during prefill
            - write only the new K/V suffix into the physical paged KV pool
            - set next_token only after the final prompt chunk
        """

        if num_tokens <= 0:
            return

        self._validate_prefill_request_state(request_state)
        input_ids = self._ensure_prompt_input_ids(request_state)

        start = request_state.num_computed_tokens
        end = min(start + num_tokens, request_state.prompt_tokens)

        if end <= start:
            return
        
        chunk_input_ids = input_ids[:, start:end]



        with torch.inference_mode():
            logits, present_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=self.model,
                input_ids = chunk_input_ids,
                past_key_values=request_state.past_key_values
            )
        
        new_past_key_values = self._slice_new_past_key_values(
            past_key_values=present_key_values,
            start_token_position=start,
            end_token_position=end
        )

        self._write_prefill_kv_to_pool(
            request_state=request_state,
            past_key_values=new_past_key_values,
            start_token_position=start
        )

        request_state.past_key_values = present_key_values
        request_state.generated_tokens = 0
        request_state.num_computed_tokens = end

        if request_state.prefill_tokens_remaining == 0:
            request_state.next_token = self._select_next_token_from_logits(logits)

            #Decode uses the physical paged KV pool, not this contiguous prefill
            # cache. Drop it after prefill completes to avoid keeping duplicate KV.

            request_state.past_key_values = None
    
    
    
    
    def init_request_state(self, request_state: RequestState) -> None:
        """
        Backward-compatible alias for full request prefill.

        New scheduler code should call prefill_request(...) to make the 
        prefill/decode lifecycle explicit.
        """
        self.prefill_request(request_state)
        
        
    
    def decode_step(
            self,
            request_states: list[RequestState],
            kv_block_manager: KVBlockManager,
    ) -> DecodeStepOutput:
        start = time.perf_counter()
        outputs: list[RequestDecodeOutput] = []

        with torch.inference_mode():
            input_id_rows: list[torch.Tensor] = []
            token_positions: list[int] = []
            block_tables: list[list[int]] = []
            text_pieces: list[str] = []
            next_token_ids: list[int] = []


            for request_state in request_states:
                if request_state.input_ids is None:
                    raise ValueError(
                        f"request_state.input_ids is None for "
                        f"request_id={request_state.request_id!r}. "
                        "Did init_request_state run?"
                    )
                if request_state.block_table is None:
                    raise ValueError(
                        f"request_state.block_table is None for "
                        f"request_id={request_state.request_id!r}. "
                        "Scheduler must allocate KV blocks before decode_step."
                    )
                if request_state.next_token is None:
                    raise ValueError(
                        f"request_state.next_token is None for "
                        f"request_id={request_state.request_id!r}. "
                        f"Did init_request_state run?"
                    )
                

                cache_seq_len = request_state.prompt_tokens + request_state.generated_tokens
                new_token_position = cache_seq_len

                next_token = request_state.next_token.to(self._device)
                next_token_id = int(next_token.item())

                request_state.input_ids = torch.cat(
                    [request_state.input_ids.to(self._device), next_token],
                    dim=-1
                )

                request_state.generated_tokens += 1

                text_piece = self.tokenizer.decode(
                    [next_token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                input_id_rows.append(next_token)
                token_positions.append(new_token_position)
                block_tables.append(request_state.block_table)
                text_pieces.append(text_piece)
                next_token_ids.append(next_token_id)
            
            batch_input_ids = torch.cat(input_id_rows, dim=0)
            block_tables_tensor = self._build_block_tables_tensor(block_tables)


            seq_lens = torch.tensor(
                [position + 1 for position in token_positions],
                device=self._device,
                dtype=torch.int32
            )
            
            logits = llama_model_decode_batch_with_paged_attention_from_hf_weights(
                hf_model=self.model,
                input_ids=batch_input_ids,
                token_positions=token_positions,
                block_tables=block_tables,
                block_tables_tensor=block_tables_tensor,
                seq_lens=seq_lens,
                kv_cache_pool=self.kv_cache_pool,
                attention_backend=self.attention_backend
            )

            for batch_index, request_state in enumerate(request_states):
                next_token_id = next_token_ids[batch_index]
                text_piece = text_pieces[batch_index]

                reached_eos = (
                    self.tokenizer.eos_token_id is not None
                    and next_token_id == self.tokenizer.eos_token_id
                )

                reached_max_new_tokens = request_state.is_finished()
                finished = bool(reached_eos or reached_max_new_tokens)

                if finished:
                    request_state.next_token = None
                else:
                    request_state.next_token = torch.argmax(
                        logits[batch_index : batch_index + 1,-1,:],
                        dim=-1,
                        keepdim=True,
                    )
                outputs.append(
                    RequestDecodeOutput(
                        request_id=request_state.request_id,
                        text=text_piece,
                        generated_tokens=1,
                        finished=finished,
                    )
                )
        backend_ms = (time.perf_counter() - start) * 1000.0
        return DecodeStepOutput(
            request_outputs=outputs,
            backend_ms=backend_ms,
            decode_batch_snapshot={
                "backend": "custom-llama-cuda-paged-attention-batched",
                "num_requests": len(request_states),
                "uses_kv_cache": True,
                "uses_kv_cache_pool": True,
                "uses_paged_attention": True,
                "attention_backend": self.attention_backend_name,
                "batched_decode": True,
                "kv_block_manager_present": kv_block_manager is not None,
            },
        )
    
    
    
   



