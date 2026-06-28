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
from engines.llama.paged_model import llama_model_decode_with_paged_attention_from_hf_weights

from engines.llama.cached_model import llama_model_forward_with_kv_cache_from_hf_weights

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"




class CustomLlamaDecodeEngine:
    """
    First custom Llama DecodeEngine.

    This version is correctness-first:

        - uses TinyLlama weights/tokenizer from Hugging Face
        - uses our custom llama_model_forward path
        - recomputes the full sequence every decode step
        - does not use KVCachePool yet
        - does not use CUDA paged attention yet

    This is intentionally inefficient. The purpose is to prove that the existing
    EngineScheduler can drive real token generation through our custom Llama math.
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
            request_states: list[RequestState]
    ) -> torch.Tensor:
        max_blocks = max(len(request_state.block_table or []) for request_state in request_states)
        if max_blocks == 0:
            raise ValueError("Cannot build block_tables tensor with no blocks")
        
        rows: list[list[int]] = []

        for request_state in request_states:
            if request_state.block_table is None:
                raise ValueError(
                    f"request_state.block_table is None for "
                    f"request_id={request_state.request_id!r}"
                )
            
            row = list(request_state.block_table)
            row = row + [-1] * (max_blocks - len(row))
            rows.append(row)
        return torch.Tensor(
            rows,
            device=self._device,
            dtype=torch.int32
        )
    
    def _single_block_table_tensor(self, block_table: list[int]) -> torch.tensor:
        return torch.tensor(
            [block_table],
            device=self._device,
            dtype=torch.int32
        )
    
    def init_request_state(self, request_state: RequestState) -> None:
        encoded = self.tokenizer(request_state.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)

        with torch.inference_mode():
            logits, past_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
                hf_model=self.model,
                input_ids=input_ids,
                past_key_values=None
            )
            next_token = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)

        if request_state.block_table is None:
            raise ValueError(
                f"request_state.block_table is None for "
                f"request_id={request_state.request_id!r}. "
                "Scheduler must allocate KV blocks before init_request_state."
            )
        
        write_past_key_values_to_pool(
            kv_cache_pool=self.kv_cache_pool,
            block_table=request_state.block_table,
            past_key_values=past_key_values,
            start_token_position=0
        )

        request_state.input_ids = input_ids
        request_state.past_key_values = None
        request_state.next_token = next_token
        request_state.prompt_tokens = int(input_ids.shape[-1])
        request_state.generated_tokens = 0
        request_state.num_computed_tokens = request_state.prompt_tokens
        
    
    def decode_step(
            self,
            request_states: list[RequestState],
            kv_block_manager: KVBlockManager,
    ) -> DecodeStepOutput:
        start = time.perf_counter()
        outputs: list[RequestDecodeOutput] = []

        with torch.inference_mode():
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

                block_tables_tensor = self._single_block_table_tensor(request_state.block_table)

                seq_lens = torch.tensor(
                    [new_token_position + 1],
                    device=self._device,
                    dtype=torch.int32
                )
                logits = llama_model_decode_with_paged_attention_from_hf_weights(
                    hf_model=self.model,
                    input_ids=next_token,
                    token_position=new_token_position,
                    block_table=request_state.block_table,
                    block_tables_tensor=block_tables_tensor,
                    seq_lens=seq_lens,
                    kv_cache_pool=self.kv_cache_pool,
                    attention_backend=self.attention_backend,
                )

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
                        logits[:,-1,:],
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
                "backend": "custom-llama-cuda-paged-attention",
                "num_requests": len(request_states),
                "uses_kv_cache": True,
                "uses_kv_cache_pool": True,
                "uses_paged_attention": True,
                "attention_backend": self.attention_backend_name,
                "kv_block_manager_present": kv_block_manager is not None,
            },
        )
    
    
    
   



