from __future__ import annotations

import inspect
import time
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from runtime.decode_engine import DecodeStepOutput, RequestDecodeOutput
from runtime.kv_block_manager import KVBlockManager
from runtime.request_state import RequestState


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
            dtype: torch.dtype | None = None
            ) -> None:
        self.model_id = model_id
        self._device = device or ("cuda" if torch.cuda.is_available else "cpu")
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

    @property
    def device(self) -> str:
        return self._device
    
    def count_prompt_tokens(self, prompt: str) -> int:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return int(input_ids.shape[-1])
    
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

        request_state.input_ids = input_ids
        request_state.past_key_values = past_key_values
        request_state.next_token = next_token
        request_state.prompt_tokens = int(input_ids.shape[-1])
        request_state.generated_tokens = 0
        
    
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
                if request_state.past_key_values is None:
                    raise ValueError(
                        f"request_state.past_key_values is None for "
                        f"request_id={request_state.request_id!r}. "
                        "Did init_request_state run?"
                    )
                if request_state.next_token is None:
                    raise ValueError(
                        f"request_state.next_token is None for "
                        f"request_id={request_state.request_id!r}. "
                        f"Did init_request_state run?"
                    )
                

                #v1 cached decode consumes the token selected by the previous
                #prefill/decode step.
                next_token = request_state.next_token.to(self._device)
                next_token_id = int(next_token.item())

                request_state.input_ids = torch.cat(
                    [request_state.input_ids.to(self._device), next_token],
                    dim=-1,
                )

                request_state.generated_tokens += 1

                text_piece = self.tokenizer.decode(
                    [next_token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
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
                    logits, present_key_values = llama_model_forward_with_kv_cache_from_hf_weights(
                        hf_model=self.model,
                        input_ids=next_token,
                        past_key_values=request_state.past_key_values
                    )
                    request_state.past_key_values = present_key_values
                    request_state.next_token = torch.argmax(
                        logits[:,-1,:],
                        dim=-1,
                        keepdim=True
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
                "backend": "custom-llama-contiguous-kv-cache",
                "num_requests": len(request_states),
                "uses_kv_cache": True,
                "uses_paged_attention": False,
                "kv_block_manager_present": kv_block_manager is not None,
            },
        )
    
    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(
            0,
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        attention_mask = build_causal_mask(
            batch_size=batch_size,
            q_len=seq_len,
            kv_len=seq_len,
            device=input_ids.device,
            dtype=self.dtype,
        )

        dummy_rope_x = torch.empty(
            batch_size,
            self.num_key_value_heads,
            seq_len,
            self.head_dim,
            device=input_ids.device,
            dtype=self.dtype,
        )

        cos, sin = self._call_hf_rotary_emb(
            rotary_emb=self.model.model.rotary_emb,
            x=dummy_rope_x,
            position_ids=position_ids,
        )

        return llama_model_forward(
            input_ids=input_ids,
            embed_tokens_weight=self.model.model.embed_tokens.weight,
            layers=list(self.model.model.layers),
            final_norm_weight=self.model.model.norm.weight,
            lm_head_weight=self.model.lm_head.weight,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            rms_norm_eps=self.config.rms_norm_eps,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )
    
    @staticmethod
    def _call_hf_rotary_emb(
        rotary_emb: torch.nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signature = inspect.signature(rotary_emb.forward)

        if "position_ids" in signature.parameters:
            return rotary_emb(x, position_ids)

        if "seq_len" in signature.parameters:
            seq_len = int(position_ids.shape[-1])
            return rotary_emb(x, seq_len=seq_len)

        return rotary_emb(x, position_ids)



