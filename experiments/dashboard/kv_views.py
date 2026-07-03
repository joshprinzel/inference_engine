from __future__ import annotations

import math

import streamlit as st
import pandas as pd

def estimate_peak_blocks(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    block_size_tokens: int,
) -> int:
    total_tokens = prompt_tokens + generated_tokens
    if total_tokens <= 0:
        return 0
    return math.ceil(total_tokens / block_size_tokens)

def estimate_multi_prompt_peak_blocks(result) -> int:
    total = 0

    for request_result in result.request_results:
        total += estimate_peak_blocks(
            prompt_tokens=request_result.prompt_tokens,
            generated_tokens=request_result.generated_tokens,
            block_size_tokens=result.block_size_tokens
        )
    return total


def get_latest_kv_result():
    multi_result = st.session_state.get("latest_multi_prompt_result")
    single_result = st.session_state.get("latest_playground_resut")

    if multi_result is not None:
        return "multi", multi_result
    
    if single_result is not None:
        return "single", single_result
    
    return None, None

def render_kv_tradeoff_explainer() -> None:
    st.subheader("Paged KV Tradeoff")

    col_small, col_large = st.columns(2)

    with col_small:
        st.markdown(
            """
            **Smaller blocks**

            - Less internal fragmentation
            - More blocks per request
            - Longer block tables
            - More paged-attention traversal pressure
            """
        )

    with col_large:
        st.markdown(
            """
            **Larger blocks**

            - Fewer blocks per request
            - Shorter block tables
            - Lower traversal pressure
            - More wasted capacity for short requests
            """
        )


def render_block_grid(
    *,
    used_blocks: int,
    total_blocks: int,
    label: str = "used",
    columns: int = 16,
) -> None:
    total_blocks = max(total_blocks, 1)
    used_blocks = max(0, min(used_blocks, total_blocks))
    columns = max(columns, 1)

    rows: list[list[str]] = []
    current_row: list[str] = []

    for block_id in range(total_blocks):
        state = label if block_id < used_blocks else "free"
        current_row.append(f"Block {block_id}\n{state}")

        if len(current_row) == columns:
            rows.append(current_row)
            current_row = []

    if current_row:
        current_row.extend([""] * (columns - len(current_row)))
        rows.append(current_row)

    df = pd.DataFrame(
        rows,
        columns=[f"slot_{index}" for index in range(columns)],
    )

    def style_cell(value: str) -> str:
        if not value:
            return "background-color: transparent; border: none;"
        if "free" in value:
            return (
                "background-color: rgba(120, 120, 120, 0.08); "
                "border: 1px solid rgba(120, 120, 120, 0.24); "
                "border-radius: 8px; "
                "text-align: center; "
                "color: rgba(255, 255, 255, 0.72); "
                "font-weight: 500;"
            )
        return (
            "background-color: rgba(30, 180, 120, 0.22); "
            "border: 1px solid rgba(30, 180, 120, 0.45); "
            "border-radius: 8px; "
            "text-align: center; "
            "color: white; "
            "font-weight: 700;"
        )

    styled = (
        df.style
        .hide(axis="index")
        .hide(axis="columns")
        .map(style_cell)
        .set_properties(
            **{
                "white-space": "pre-line",
                "min-width": "70px",
                "height": "54px",
                "padding": "0.45rem",
            }
        )
    )

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True
    )
def render_latest_run_kv_view(result) -> None:
    st.subheader("Latest Playground Run KV Usage")

    expected_peak_blocks = estimate_peak_blocks(
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.tokens_generated,
        block_size_tokens=result.block_size_tokens,
    )
    observed_peak_blocks = int(result.kv_peak_used_blocks)
    final_blocks = int(result.kv_final_used_blocks)

    blocks_match = expected_peak_blocks == observed_peak_blocks

    cols = st.columns(6)
    cols[0].metric("Observed peak", str(observed_peak_blocks))
    cols[1].metric("Expected peak", str(expected_peak_blocks))
    cols[2].metric("Final KV blocks", str(final_blocks))
    cols[3].metric("Generated tokens", str(result.tokens_generated))
    cols[4].metric("Prompt tokens", str(result.prompt_tokens))
    cols[5].metric(
        "Reclaimed",
        "YES" if final_blocks == 0 and observed_peak_blocks > 0 else "NO",
    )

    st.markdown("### Peak allocation")
    st.caption(
        "This visual shows the maximum number of KV blocks used during the latest run."
    )

    visual_total = max(16, min(result.total_kv_blocks, max(observed_peak_blocks * 2, expected_peak_blocks * 2, 16)))

    render_block_grid(
        used_blocks=observed_peak_blocks,
        total_blocks=visual_total,
        label="peak",
        columns=8 if visual_total <= 32 else 16,
    )

    st.markdown("### Final allocation")
    st.caption(
        "After the request finishes, blocks should be returned to the KV block manager."
    )

    render_block_grid(
        used_blocks=final_blocks,
        total_blocks=visual_total,
        label="used",
        columns=8 if visual_total <= 32 else 16,
    )

    with st.expander("KV details"):
        st.json(
            {
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.tokens_generated,
                "block_size_tokens": result.block_size_tokens,
                "total_kv_blocks": result.total_kv_blocks,
                "expected_peak_blocks": expected_peak_blocks,
                "observed_peak_blocks": observed_peak_blocks,
                "expected_matches_observed": blocks_match,
                "kv_final_used_blocks": result.kv_final_used_blocks,
                "kv_final_free_blocks": result.kv_final_free_blocks,
                "decode_iterations": result.decode_iterations,
                "last_decode_batch": result.last_decode_batch,
            }
        )

def render_latest_multi_prompt_kv_view(result) -> None:
    st.subheader("Latest Multi-Prompt KV Usage")

    expected_peak_blocks = estimate_multi_prompt_peak_blocks(result)
    observed_peak_blocks = int(result.kv_peak_used_blocks)
    final_blocks = int(result.kv_final_used_blocks)
    blocks_match = expected_peak_blocks == observed_peak_blocks

    cols = st.columns(6)
    cols[0].metric("Observed peak", str(observed_peak_blocks))
    cols[1].metric("Expected peak", str(expected_peak_blocks))
    cols[2].metric("Final KV blocks", str(final_blocks))
    cols[3].metric("Generated tokens", str(result.tokens_generated))
    cols[4].metric("Requests", str(len(result.request_results)))
    cols[5].metric(
        "Reclaimed",
        "YES" if final_blocks == 0 and observed_peak_blocks > 0 else "NO",
    )

    if blocks_match and final_blocks == 0:
        st.success(
            "KV cache check passed: observed peak matches expected multi-request allocation and all blocks were reclaimed."
        )
    elif not blocks_match:
        st.warning(
            "Observed KV peak does not match expected multi-request block usage. Inspect allocation or token accounting."
        )
    else:
        st.warning(
            "KV blocks were not fully reclaimed after request completion."
        )

    st.markdown("### Peak allocation")

    visual_total = max(
        16,
        min(
            result.total_kv_blocks,
            max(observed_peak_blocks * 2, expected_peak_blocks * 2, 16),
        ),
    )

    render_block_grid(
        used_blocks=observed_peak_blocks,
        total_blocks=visual_total,
        label="peak",
        columns=8 if visual_total <= 32 else 16,
    )

    st.markdown("### Final allocation after completion")

    render_block_grid(
        used_blocks=final_blocks,
        total_blocks=visual_total,
        label="used",
        columns=8 if visual_total <= 32 else 16,
    )

    st.markdown("### Per-request KV estimate")

    rows = []

    for request_result in result.request_results:
        rows.append(
            {
                "request_id": request_result.request_id,
                "prompt_tokens": request_result.prompt_tokens,
                "generated_tokens": request_result.generated_tokens,
                "total_tokens": request_result.prompt_tokens
                + request_result.generated_tokens,
                "expected_blocks": estimate_peak_blocks(
                    prompt_tokens=request_result.prompt_tokens,
                    generated_tokens=request_result.generated_tokens,
                    block_size_tokens=result.block_size_tokens,
                ),
                "status": request_result.final_status,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("KV details"):
        st.json(
            {
                "mode": "multi_prompt",
                "request_count": len(result.request_results),
                "block_size_tokens": result.block_size_tokens,
                "total_kv_blocks": result.total_kv_blocks,
                "expected_peak_blocks": expected_peak_blocks,
                "observed_peak_blocks": observed_peak_blocks,
                "expected_matches_observed": blocks_match,
                "kv_final_used_blocks": result.kv_final_used_blocks,
                "kv_final_free_blocks": result.kv_final_free_blocks,
                "decode_iterations": result.decode_iterations,
                "last_decode_batch": result.last_decode_batch,
            }
        )


def render_kv_cache_inspector() -> None:
    st.header("KV Cache Inspector")

    st.markdown(
        """
        This page visualizes how the runtime uses paged KV cache blocks during the
        latest playground run. It compares expected block usage against observed
        scheduler telemetry and checks whether blocks were reclaimed after completion.
        """
    )

    result_mode, result = get_latest_kv_result()

    if result is None:
        st.info("Run a prompt or multi-prompt batch in the Runtime Playground to populate the KV inspector.")
        render_kv_tradeoff_explainer()
        return

    if result_mode == "multi":
        render_latest_multi_prompt_kv_view(result)
    else:
        render_latest_run_kv_view(result)

    render_kv_tradeoff_explainer()