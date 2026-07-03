from __future__ import annotations

import torch
import streamlit as st

from experiments.dashboard.runtime_playground import (
    MultiPromptPlaygroundResult,
    PlaygroundResult,
    create_playground_engine,
    run_tinyllama_multi_prompt_with_engine,
    run_tinyllama_request_with_engine,
)
from experiments.dashboard.styles import render_status_badges


PRESET_PROMPTS = {
    "Capital: France": "The capital of France is",
    "Capital: Germany": "The capital of Germany is",
    "Cache explanation": "In computer science, a cache is",
    "GPU explanation": "A GPU is useful for",
    "Story Starter": "Once upon a time"
}

MULTI_PROMPT_PRESETS = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
]


def render_multi_prompt_metric_cards(result: MultiPromptPlaygroundResult) -> None:
    cols = st.columns(6)
    cols[0].metric("Tokens/sec", f"{result.tokens_per_second:.2f}")
    cols[1].metric("Generated tokens", str(result.tokens_generated))
    cols[2].metric("Backend median", f"{result.backend_ms_median:.3f} ms")
    cols[3].metric("Backend p95", f"{result.backend_ms_p95:.3f} ms")
    cols[4].metric("Decode steps", str(result.decode_iterations))
    cols[5].metric("Peak KV blocks", str(result.kv_peak_used_blocks))


def render_multi_prompt_results(result: MultiPromptPlaygroundResult) -> None:
    render_multi_prompt_metric_cards(result)

    last_decode_batch = result.last_decode_batch or {}

    render_status_badges(
        cuda_paged_attention=bool(last_decode_batch.get("uses_paged_attention", False)),
        kv_cache_pool=bool(last_decode_batch.get("uses_kv_cache_pool", False)),
        batched_decode=bool(last_decode_batch.get("batched_decode", False)),
        correctness=result.all_finished,
    )

    st.subheader("Per-request outputs")

    for request_result in result.request_results:
        with st.expander(
            f"{request_result.request_id} · status={request_result.final_status}",
            expanded=True,
        ):
            st.markdown("**Prompt**")
            st.code(request_result.prompt)

            st.markdown("**Generated text**")
            st.code(request_result.generated_text)

            cols = st.columns(3)
            cols[0].metric("Prompt tokens", str(request_result.prompt_tokens))
            cols[1].metric("Generated tokens", str(request_result.generated_tokens))
            cols[2].metric("Error", request_result.error or "None")

    with st.expander("Multi-prompt runtime details"):
        st.json(
            {
                "max_new_tokens": result.max_new_tokens,
                "block_size_tokens": result.block_size_tokens,
                "total_kv_blocks": result.total_kv_blocks,
                "max_slots": result.max_slots,
                "tokens_generated": result.tokens_generated,
                "decode_iterations": result.decode_iterations,
                "decode_batches_built": result.decode_batches_built,
                "kv_peak_used_blocks": result.kv_peak_used_blocks,
                "kv_final_used_blocks": result.kv_final_used_blocks,
                "kv_final_free_blocks": result.kv_final_free_blocks,
                "all_finished": result.all_finished,
                "last_decode_batch": result.last_decode_batch,
                "step_trace": result.step_trace,
            }
        )
def parse_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    
    raise ValueError(f"Unsupported dtype: {dtype_name}")



def render_playground_controls() -> dict:
    st.sidebar.header("Runtime Controls")

    max_new_tokens = st.sidebar.slider(
        "Max new tokens",
        min_value=1,
        max_value=128,
        value=32,
        step=1
    )

    block_size_tokens = st.sidebar.selectbox(
        "Block size tokens",
        options=[4,8,16,32],
        index=2
    )

    total_kv_blocks = st.sidebar.number_input(
        "Total KV blocks",
        min_value=16,
        max_value=4096,
        value=256,
        step=16
    )

    max_slots = st.sidebar.number_input(
        "Max slots",
        min_value=1,
        max_value=16,
        value=1,
        step=1
    )

    dtype_name = st.sidebar.selectbox(
        "DType",
        options=["float16","bfloat16","float32"],
        index=0,
    )

    device = st.sidebar.selectbox(
        "Device",
        options=["cuda", "cpu"],
        index=0
    )

    attention_backend_name = st.sidebar.selectbox(
        "Attention backend",
        options=["cuda","reference"],
        index=0
    )

    return {
        "max_new_tokens": int(max_new_tokens),
        "block_size_tokens": int(block_size_tokens),
        "total_kv_blocks": int(total_kv_blocks),
        "max_slots": int(max_slots),
        "dtype": parse_dtype(dtype_name),
        "dtype_name": dtype_name,
        "device": device,
        "attention_backend_name": attention_backend_name
    }

def render_prompt_preset_controls() -> str | None:
    st.sidebar.header("Prompt Presets")

    preset_name = st.sidebar.selectbox(
        "Preset prompt",
        options=["Custom"] + list(PRESET_PROMPTS.keys()),
    )
    if preset_name == "Custom":
        return None
    
    preset_prompt = PRESET_PROMPTS[preset_name]
    st.sidebar.caption(f"Preset: {preset_prompt}")
    return preset_prompt

def render_metric_cards(result: PlaygroundResult) -> None:
    cols = st.columns(6)

    cols[0].metric("Tokens/sec",f"{result.tokens_per_second:.2f}")
    cols[1].metric("Generated Tokens", str(result.tokens_generated))
    cols[2].metric("Backend median", f"{result.backend_ms_median:.3f} ms")
    cols[3].metric("Backend p95", f"{result.backend_ms_p95:.3f} ms")
    cols[4].metric("Decode steps", str(result.decode_iterations))
    cols[5].metric("Peak KV Blocks", str(result.kv_peak_used_blocks))


def infer_runtime_flags(result: PlaygroundResult) -> tuple[bool,bool,bool]:
    last_decode_batch = result.last_decode_batch or {}

    cuda_paged_attention = bool(last_decode_batch.get("uses_paged_attention", False))
    kv_cache_pool = bool(last_decode_batch.get("uses_kv_cache_pool", False))
    batched_decode = bool(last_decode_batch.get("batched_decode", False))

    return cuda_paged_attention, kv_cache_pool, batched_decode


def render_result_details(result: PlaygroundResult) -> None:
    cuda_paged_attention, kv_cache_pool, batched_decode = infer_runtime_flags(result)

    render_status_badges(
        cuda_paged_attention=cuda_paged_attention,
        kv_cache_pool=kv_cache_pool,
        batched_decode=batched_decode,
        correctness=result.final_status == "finished" and result.error is None
    )
    render_metric_cards(result)

    with st.expander("Runtime details", expanded=False):
        st.json(
            {
                "prompt": result.prompt,
                "generated_text": result.generated_text,
                "full_text": result.full_text,
                "final_status": result.final_status,
                "error": result.error,
                "max_new_tokens": result.max_new_tokens,
                "tokens_generated": result.tokens_generated,
                "tokens_per_second": result.tokens_per_second,
                "total_wall_seconds": result.total_wall_seconds,
                "decode_iterations": result.decode_iterations,
                "decode_batches_built": result.decode_batches_built,
                "backend_ms_median": result.backend_ms_median,
                "backend_ms_p95": result.backend_ms_p95,
                "backend_ms_min": result.backend_ms_min,
                "backend_ms_max": result.backend_ms_max,
                "kv_peak_used_blocks": result.kv_peak_used_blocks,
                "kv_final_used_blocks": result.kv_final_used_blocks,
                "kv_final_free_blocks": result.kv_final_free_blocks,
                "last_decode_batch": result.last_decode_batch,
                
            }
        )

def append_message(role: str, content: str, result: PlaygroundResult | None = None) -> None:
    st.session_state.playground_messages.append(
        {
            "role": role,
            "content": content,
            "result": result
        }
    )


def render_existing_messages() -> None:
    for message in st.session_state.playground_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            result = message.get("result")
            if result is not None:
                render_result_details(result)

@st.cache_resource(show_spinner="Loading TinyLlama runtime...")
def get_cached_engine(
    *,
    block_size_tokens:int,
    total_kv_blocks: int,
    dtype_name: str,
    device: str,
    attention_backend_name: str
):
    return create_playground_engine(
        block_size_tokens=block_size_tokens,
        total_kv_blocks=total_kv_blocks,
        dtype=parse_dtype(dtype_name),
        device=device,
        attention_backend_name=attention_backend_name
    )

def render_runtime_playground() -> None:
    st.header("Runtime Playground")

    st.markdown(
        """
        Send prompts through the custom TinyLlama serving path. This does not call
        Hugging Face `generate`; it exercises the scheduler, KV block manager,
        KV cache pool, and CUDA paged-attention backend.

        This is currently completion-style generation, not TinyLlama chat-template
        generation, so short instruction prompts may produce odd continuations.
        """
    )

    controls = render_playground_controls()
    preset_prompt = render_prompt_preset_controls()

    if st.sidebar.button("Reload TinyLlama runtime"):
        get_cached_engine.clear()
        st.rerun()

    st.sidebar.caption(
        "Changing block size, KV blocks, dtype, device, or attention backend "
        "reloads the cached runtime."
    )

    engine = get_cached_engine(
        block_size_tokens=controls["block_size_tokens"],
        total_kv_blocks=controls["total_kv_blocks"],
        dtype_name=controls["dtype_name"],
        device=controls["device"],
        attention_backend_name=controls["attention_backend_name"],
    )

    single_tab, multi_tab = st.tabs(["Single Prompt", "Multi-Prompt Batch"])

    with single_tab:
        if "playground_messages" not in st.session_state:
            st.session_state.playground_messages = []

        if st.button("Clear conversation", key="clear_single_conversation"):
            st.session_state.playground_messages = []
            st.rerun()

        render_existing_messages()

        prompt = None

        if preset_prompt is not None:
            if st.button("Run preset prompt", key="run_single_preset"):
                prompt = preset_prompt

        chat_prompt = st.chat_input("Prompt TinyLlama through your runtime...")

        if chat_prompt is not None:
            prompt = chat_prompt

        if prompt is not None:
            append_message("user", prompt)

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.status(
                    "Running TinyLlama through custom runtime...",
                    expanded=False,
                ):
                    try:
                        result = run_tinyllama_request_with_engine(
                            engine=engine,
                            prompt=prompt,
                            max_new_tokens=controls["max_new_tokens"],
                            block_size_tokens=controls["block_size_tokens"],
                            total_kv_blocks=controls["total_kv_blocks"],
                            max_slots=controls["max_slots"],
                            device=controls["device"],
                        )
                    except Exception as exc:
                        st.error(f"Runtime failed: {exc!r}")
                        append_message("assistant", f"Runtime failed: `{exc!r}`")
                        return

                st.markdown(result.generated_text)
                render_result_details(result)

            append_message("assistant", result.generated_text, result=result)
            st.session_state.latest_playground_result = result

    with multi_tab:
        st.subheader("Multi-Prompt Batch Playground")

        st.markdown(
            """
            Run several prompts through one scheduler workload. This demonstrates
            continuous batching at the runtime level and batched decode inside the engine.
            """
        )

        prompt_count = st.slider(
            "Number of prompts",
            min_value=2,
            max_value=4,
            value=4,
            key="multi_prompt_count",
        )

        prompts: list[str] = []

        for index in range(prompt_count):
            default_prompt = MULTI_PROMPT_PRESETS[index]
            prompt_value = st.text_area(
                f"Prompt {index + 1}",
                value=default_prompt,
                height=80,
                key=f"multi_prompt_{index}",
            )
            prompts.append(prompt_value)

        if st.button("Run multi-prompt batch", key="run_multi_prompt_batch"):
            with st.status(
                "Running multi-prompt batch through custom runtime...",
                expanded=False,
            ):
                try:
                    result = run_tinyllama_multi_prompt_with_engine(
                        engine=engine,
                        prompts=prompts,
                        max_new_tokens=controls["max_new_tokens"],
                        block_size_tokens=controls["block_size_tokens"],
                        total_kv_blocks=controls["total_kv_blocks"],
                        max_slots=max(controls["max_slots"], prompt_count),
                        device=controls["device"],
                    )
                except Exception as exc:
                    st.error(f"Multi-prompt runtime failed: {exc!r}")
                    st.exception(exc)
                    return

            st.session_state.latest_multi_prompt_result = result
            render_multi_prompt_results(result)

        latest_multi = st.session_state.get("latest_multi_prompt_result")

        if latest_multi is not None:
            st.markdown("---")
            st.subheader("Latest multi-prompt result")
            render_multi_prompt_results(latest_multi)