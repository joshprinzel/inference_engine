import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from threading import Thread


class ModelRunner:
    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
            device: str | None = None
    ):
        self.model_name = model_name
        self.device = device or self.__default_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()
    

    def __default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens:int) -> dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        prompt_tokens = inputs["input_ids"].shape[-1]

        self.synchronize()
        start = time.perf_counter()

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        self.synchronize()
        end = time.perf_counter()

        total_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        generated_ids = output_ids[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        generated_tokens = len(generated_ids)
        latency_seconds = end - start


        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "total_text": total_text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "latency_seconds": latency_seconds,
            "tokens_per_second": (
                generated_tokens / latency_seconds
                if latency_seconds > 0
                else 0.0
            ),
        }
    
    @torch.inference_mode()
    def stream_generate(self, prompt: str, max_new_tokens: int):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30.0,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )
        thread.start()

        for text in streamer:
            yield text

        thread.join(timeout=1.0)
    

    @torch.inference_mode()
    def prefill(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        prompt_tokens = inputs["input_ids"].shape[-1]

        self.synchronize()
        start = time.perf_counter()

        outputs = self.model(
            **inputs,
            use_cache=True
        )
        self.synchronize()
        end = time.perf_counter()

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "past_key_values": outputs.past_key_values,
            "logits": outputs.logits,
            "prompt_tokens": prompt_tokens,
            "prefill_time_seconds": end - start,
            }

    @torch.inference_mode()
    def decode_from_prefill(
        self,
        prefill_result: dict,
        max_new_tokens: int,
    ) -> dict:
        past_key_values = prefill_result["past_key_values"]
        next_token = torch.argmax(
            prefill_result["logits"][:, -1, :],
            dim=-1,
            keepdim=True
        )

        generated_tokens = 0

        self.synchronize()
        start = time.perf_counter()

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True, 
            )

            past_key_values = outputs.past_key_values
            next_token = torch.argmax(
                outputs.logits[:,-1,:],
                dim=-1,
                keepdim=True
            )
            generated_tokens += 1
        
        self.synchronize()
        end = time.perf_counter()

        decode_time_seconds = end - start

        return{
            "generated_tokens": generated_tokens,
            "decode_time_seconds": decode_time_seconds,
            "decode_tokens_per_second": (
                generated_tokens / decode_time_seconds
                if decode_time_seconds > 0
                else 0.0
            ),
        }
    
    @torch.inference_mode()
    def init_request_state(self, request_state) -> None:
        prefill_result = self.prefill(request_state.prompt)

        request_state.input_ids = prefill_result["input_ids"]
        request_state.attention_mask = prefill_result["attention_mask"]
        request_state.past_key_values = prefill_result["past_key_values"]
        request_state.prompt_tokens = prefill_result["prompt_tokens"]

        request_state.next_token = torch.argmax(
            prefill_result["logits"][:,-1,:],
            dim=-1,
            keepdim=True,
        )
        request_state.mark_decoding()

    @torch.inference_mode()
    def decode_one_token(self, request_state) -> str:
        token_to_emit = request_state.next_token

        text = self.tokenizer.decode(
            token_to_emit[0],
            skip_special_tokens=True
        )

        outputs = self.model(
            input_ids=token_to_emit,
            past_key_values=request_state.past_key_values,
            use_cache=True,
        )

        request_state.past_key_values = outputs.past_key_values
        request_state.next_token = torch.argmax(
            outputs.logits[:,-1,:],
            dim=-1,
            keepdim=True
        )
        request_state.generated_tokens += 1
        return text
    

    def count_prompt_tokens(self, prompt: str) -> int:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return int(inputs["input_ids"].shape[-1])

    
    def benchmark_prefill_decode(
            self,
            prompt: str,
            max_new_tokens: int,
        ) -> dict:
        prefill_result = self.prefill(prompt)
        decode_result = self.decode_from_prefill(
            prefill_result=prefill_result,
            max_new_tokens=max_new_tokens,
        )

        total_time = (
            prefill_result["prefill_time_seconds"]
            + decode_result["decode_time_seconds"]
        )

        return {
        "prompt_tokens": prefill_result["prompt_tokens"],
        "generated_tokens": decode_result["generated_tokens"],
        "prefill_time_seconds": prefill_result["prefill_time_seconds"],
        "decode_time_seconds": decode_result["decode_time_seconds"],
        "total_time_seconds": total_time,
        "decode_tokens_per_second": decode_result["decode_tokens_per_second"],
        "total_tokens_per_second": (
            decode_result["generated_tokens"] / total_time
            if total_time > 0
            else 0.0
            ),
        }
    

    @torch.inference_mode()
    def init_batch_request_states(self, request_states: list) -> dict:
        prompts = [request_state.prompt for request_state in request_states]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        prompt_tokens_per_request = inputs["attention_mask"].sum(dim=-1).tolist()
        self.synchronize()
        outputs = self.model(
            **inputs,
            use_cache=True
        )
        self.synchronize()

        next_tokens = torch.argmax(
            outputs.logits[:,-1,:],
            dim=-1,
            keepdim=True
        )

        for index, request_state in enumerate(request_states):
            request_state.prompt_tokens = int(prompt_tokens_per_request[index])
            request_state.mark_decoding()

        return{
            "past_key_values": outputs.past_key_values,
            "next_tokens": next_tokens,
            "attention_mask": inputs["attention_mask"],
        }
    
    @torch.inference_mode()
    def decode_one_token_batch(
        self,
        request_states: list,
        batch_state: dict,
    ) -> dict:
        next_tokens = batch_state["next_tokens"]

        texts: list[str] = []

        for index, request_state in enumerate(request_states):
            token = next_tokens[index]
            text = self.tokenizer.decode(
                token,
                skip_special_tokens=True
            )
            texts.append(text)
        
        #Extend attention mask by one real token for this decode step
        old_attention_mask = batch_state["attention_mask"]
        new_token_mask = torch.ones(
            (old_attention_mask.shape[0],1),
            dtype=old_attention_mask.dtype,
            device=old_attention_mask.device,
        )

        attention_mask = torch.cat(
            [old_attention_mask, new_token_mask],
            dim=-1
        )
        
        self.synchronize()
        outputs = self.model(
            input_ids=next_tokens,
            attention_mask=attention_mask,
            past_key_values=batch_state["past_key_values"],
            use_cache=True
        )
        self.synchronize()

        new_next_tokens = torch.argmax(
            outputs.logits[:,-1,:],
            dim=-1,
            keepdim=True,
        )
        return {
            "texts": texts,
            "past_key_values": outputs.past_key_values,
            "next_tokens": new_next_tokens,
            "attention_mask": attention_mask,
        }