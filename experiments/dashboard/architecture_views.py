from __future__ import annotations

import pandas as pd
import streamlit as st

def render_runtime_path() -> None:
    st.subheader("Runtime Path")

    st.markdown(
        """
        ```text
        User Prompt
          ↓
        RequestQueue
          ↓
        EngineScheduler
          ├── admission control
          ├── KV block allocation
          ├── decode batch construction
          └── request completion / KV reclamation
          ↓
        CustomLlamaDecodeEngine
          ├── TinyLlama token selection
          ├── batched decode over active requests
          └── per-request output update
          ↓
        KVCachePool
          ├── physical key cache
          └── physical value cache
          ↓
        CUDA Paged Attention Backend
          └── gather K/V by block table and run decode attention
          ↓
        Generated Text + Runtime Metrics
        ```
        """
    )

def render_latest_run_summary(result) -> None:
    st.subheader("Latest Playground Run")

    cols = st.columns(6)
    cols[0].metric("Tokens/sec", f"{result.tokens_per_second:.2f}")
    cols[1].metric("Generated tokens", str(result.tokens_generated))
    cols[2].metric("Decode steps", str(result.decode_iterations))
    cols[3].metric("Decode batches", str(result.decode_batches_built))
    cols[4].metric("Peak KV blocks", str(result.kv_peak_used_blocks))
    cols[5].metric("Final KV blocks", str(result.kv_final_used_blocks))

    st.markdown("**Prompt**")
    st.code(result.prompt)

    st.markdown("**Generated Text**")
    st.code(result.generated_text)

    with st.expander("Last decode batch"):
        st.json(result.last_decode_batch)

def render_latest_multi_prompt_summary(result) -> None:
    st.subheader("Latest Multi-Prompt Batch Run")

    cols = st.columns(6)
    cols[0].metric("Tokens/sec", f"{result.tokens_per_second:.2f}")
    cols[1].metric("Generated tokens", str(result.tokens_generated))
    cols[2].metric("Decode steps", str(result.decode_iterations))
    cols[3].metric("Decode batches", str(result.decode_batches_built))
    cols[4].metric("Peak KV blocks", str(result.kv_peak_used_blocks))
    cols[5].metric("Final KV blocks", str(result.kv_final_used_blocks))

    last_decode_batch = result.last_decode_batch or {}

    badge_cols = st.columns(4)
    badge_cols[0].metric(
        "Batched decode",
        "ON" if last_decode_batch.get("batched_decode", False) else "OFF",
    )
    badge_cols[1].metric(
        "Paged attention",
        "ON" if last_decode_batch.get("uses_paged_attention", False) else "OFF",
    )
    badge_cols[2].metric(
        "KV cache pool",
        "ON" if last_decode_batch.get("uses_kv_cache_pool", False) else "OFF",
    )
    badge_cols[3].metric(
        "All finished",
        "YES" if result.all_finished else "NO",
    )

    st.markdown("**Per-request outputs**")

    for request_result in result.request_results:
        with st.expander(
            f"{request_result.request_id} · status={request_result.final_status}",
            expanded=False,
        ):
            st.markdown("**Prompt**")
            st.code(request_result.prompt)

            st.markdown("**Generated text**")
            st.code(request_result.generated_text)

            cols = st.columns(3)
            cols[0].metric("Prompt tokens", str(request_result.prompt_tokens))
            cols[1].metric("Generated tokens", str(request_result.generated_tokens))
            cols[2].metric("Error", request_result.error or "None")

    with st.expander("Last decode batch"):
        st.json(result.last_decode_batch)

def render_step_trace(result) -> None:
    st.subheader("Scheduler Step Trace")

    if not result.step_trace:
        st.info("No step trace available for the latest run.")
        return
    
    trace_df = pd.DataFrame(result.step_trace)

    st.markdown(
        """
        The trace shows how the scheduler state changes over time. Each row is one
        scheduler step.
        """
    )

    st.dataframe(trace_df, use_container_width=True)
    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("### Tokens Generated Over Time")
        st.line_chart(trace_df, x="step", y="tokens_generated")
    
    with chart_cols[1]:
        st.markdown("### KV Blocks Used Over Time")
        st.line_chart(trace_df, x="step", y="kv_used_blocks")
    

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("### Backend Latency Per Step")
        st.line_chart(trace_df, x="step", y="last_backend_ms")

    with chart_cols[1]:
        st.markdown("### Decode Batch Size Per Step")
        st.line_chart(trace_df, x="step", y="decode_batch_size")


def render_architecture_trace() -> None:
    st.header("Architecture Trace")

    st.markdown(
        """
        This page explains how prompts move through the runtime and visualizes the
        latest playground run as scheduler-step telemetry.
        """
    )

    render_runtime_path()

    multi_result = st.session_state.get("latest_multi_prompt_result")
    single_result = st.session_state.get("latest_playground_result")

    if multi_result is None and single_result is None:
        st.info("Run a prompt in the Runtime Playground to populate the live trace.")
        return

    result_mode = st.radio(
        "Trace source",
        options=["Latest multi-prompt batch", "Latest single prompt"],
        index=0 if multi_result is not None else 1,
        horizontal=True,
    )

    if result_mode == "Latest multi-prompt batch":
        if multi_result is None:
            st.info("No multi-prompt batch run is available yet.")
            return

        render_latest_multi_prompt_summary(multi_result)
        render_step_trace(multi_result)
        return

    if single_result is None:
        st.info("No single-prompt run is available yet.")
        return

    render_latest_run_summary(single_result)
    render_step_trace(single_result)