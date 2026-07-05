from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from experiments.dashboard.benchmark_data import (
    DEFAULT_BENCHMARK_DIR,
    aggregate_medians,
    filter_measured_rows,
    list_benchmark_csvs,
    load_benchmark_csvs,
)


def render_benchmark_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Benchmark Filters")

    filtered = df.copy()

    if "run_kind" in filtered.columns:
        run_kinds = sorted(filtered["run_kind"].dropna().unique().tolist())
        default = ["measured"] if "measured" in run_kinds else run_kinds

        selected_run_kinds = st.sidebar.multiselect(
            "Run kind",
            options=run_kinds,
            default=default,
        )

        filtered = filtered[filtered["run_kind"].isin(selected_run_kinds)]

    filter_columns = [
        "matrix_run",
        "scenario_name",
        "policy_name",
        "policy_dir",
        "backend",
        "num_requests",
        "max_slots",
        "max_new_tokens",
        "block_size_tokens",
        "dtype",
        "device",
        "prompt_set",
    ]

    for column in filter_columns:
        if column not in filtered.columns:
            continue

        values = sorted(filtered[column].dropna().unique().tolist())

        if not values:
            continue

        selected_values = st.sidebar.multiselect(
            column,
            options=values,
            default=values,
        )

        filtered = filtered[filtered[column].isin(selected_values)]

    return filtered


def render_benchmark_metric_cards(df: pd.DataFrame) -> None:
    st.subheader("Benchmark Summary")

    cols = st.columns(8)

    if df.empty:
        for col in cols:
            col.metric("No data", "—")
        return

    correctness_rate = 0.0
    if "correctness_passed" in df.columns:
        correctness_rate = 100.0 * float(df["correctness_passed"].mean())

    all_finished_rate = 0.0
    if "all_finished" in df.columns:
        all_finished_rate = 100.0 * float(df["all_finished"].mean())

    cols[0].metric("Rows", str(len(df)))
    cols[1].metric("Median tok/s", f"{df['tokens_per_second'].median():.2f}")
    cols[2].metric("Median TTFT", f"{df['avg_ttft_ms'].median():.3f} ms")
    cols[3].metric("Median latency", f"{df['avg_latency_ms'].median():.3f} ms")
    cols[4].metric("Median backend", f"{df['backend_ms_median'].median():.3f} ms")
    cols[5].metric("Decode batches", f"{df['decode_batches_built'].median():.0f}")
    cols[6].metric("Finished", f"{all_finished_rate:.1f}%")
    cols[7].metric("Correctness", f"{correctness_rate:.1f}%")

def render_line_chart(
    df: pd.DataFrame,
    *,
    title: str,
    x: str,
    y: str,
    color: str | None = None,
) -> None:
    st.markdown(f"### {title}")

    if df.empty:
        st.info("No rows match the selected filters.")
        return

    if x not in df.columns or y not in df.columns:
        st.info(f"Missing columns: `{x}` or `{y}`.")
        return

    group_columns = [x]
    if color is not None and color in df.columns:
        group_columns.append(color)

    plot_df = aggregate_medians(df, group_columns=group_columns)

    if plot_df.empty:
        st.info("No plot data available.")
        return

    if color is not None and color in plot_df.columns:
        st.line_chart(plot_df, x=x, y=y, color=color)
    else:
        st.line_chart(plot_df, x=x, y=y)

    with st.expander(f"Data: {title}"):
        st.dataframe(plot_df, use_container_width=True)


def render_concurrency_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        These plots show whether throughput and latency improve as more requests
        are active at the same time.
        """
    )

    render_line_chart(
        df,
        title="Throughput scaling after batched decode",
        x="num_requests",
        y="tokens_per_second",
        color="max_new_tokens",
    )

    render_line_chart(
        df,
        title="Backend median latency vs request count",
        x="num_requests",
        y="backend_ms_median",
        color="max_new_tokens",
    )

    render_line_chart(
        df,
        title="Backend p95 latency vs request count",
        x="num_requests",
        y="backend_ms_p95",
        color="max_new_tokens",
    )

    render_line_chart(
        df,
        title="KV block pressure vs request count",
        x="num_requests",
        y="kv_peak_used_blocks",
        color="max_new_tokens",
    )


def render_block_size_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        These plots show the paged-KV block-size tradeoff. Smaller blocks can reduce
        internal fragmentation but increase block-table pressure. Larger blocks reduce
        metadata pressure but may waste capacity for short requests.
        """
    )

    render_line_chart(
        df,
        title="Throughput vs KV block size",
        x="block_size_tokens",
        y="tokens_per_second",
        color="max_new_tokens",
    )

    render_line_chart(
        df,
        title="Backend median latency vs KV block size",
        x="block_size_tokens",
        y="backend_ms_median",
        color="max_new_tokens",
    )

    render_line_chart(
        df,
        title="Backend p95 latency vs KV block size",
        x="block_size_tokens",
        y="backend_ms_p95",
        color="max_new_tokens",
    )

    render_line_chart(
        df,
        title="Peak KV blocks vs KV block size",
        x="block_size_tokens",
        y="kv_peak_used_blocks",
        color="max_new_tokens",
    )


def render_latency_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        Latency metrics include Python orchestration, PyTorch model work, and the
        custom CUDA paged-attention backend inside `decode_step`.
        """
    )

    latency_columns = [
        "avg_queue_wait_ms",
        "avg_ttft_ms",
        "avg_decode_latency_ms",
        "avg_latency_ms",
        "backend_ms_median",
        "backend_ms_p95",
        "backend_ms_min",
        "backend_ms_max",
        "backend_ms_mean",
    ]

    available = [column for column in latency_columns if column in df.columns]

    if not available:
        st.info("No latency columns found.")
        return

    display_columns = [
        "scenario_name",
        "policy_name",
        "backend",
        "num_requests",
        "max_slots",
        "max_new_tokens",
        "block_size_tokens",
        "run_kind",
        *available,
    ]

    display_columns = [column for column in display_columns if column in df.columns]

    sort_columns = [
        column
        for column in ["scenario_name", "policy_name", "num_requests", "max_new_tokens", "block_size_tokens"]
        if column in df.columns
    ]

    st.dataframe(
        df[display_columns].sort_values(sort_columns, ascending=True),
        use_container_width=True,
    )


def render_raw_data_tab(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True)

def render_policy_comparison_tab(df: pd.DataFrame) -> None:
    st.markdown(
        """
        Compare scheduler policies across benchmark scenarios. Rows are grouped
        by scenario and policy, using median values across measured repeats.
        """
    )

    measured_df = filter_measured_rows(df)

    if measured_df.empty:
        st.info("No measured benchmark rows available.")
        return

    group_columns = [
        column
        for column in ["scenario_name", "policy_name", "prompt_set", "num_requests", "max_slots", "max_new_tokens"]
        if column in measured_df.columns
    ]

    if not group_columns:
        st.info("No grouping columns available for policy comparison.")
        return

    comparison_df = aggregate_medians(
        measured_df,
        group_columns=group_columns,
    )

    if comparison_df.empty:
        st.info("No comparison data available.")
        return

    display_columns = [
        "scenario_name",
        "policy_name",
        "prompt_set",
        "num_requests",
        "max_slots",
        "max_new_tokens",
        "tokens_per_second",
        "avg_ttft_ms",
        "avg_latency_ms",
        "avg_queue_wait_ms",
        "avg_decode_latency_ms",
        "decode_batches_built",
        "backend_ms_median",
        "backend_ms_p95",
        "kv_peak_used_blocks",
        "all_finished_all_passed",
        "correctness_all_passed",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in comparison_df.columns
    ]

    st.dataframe(
        comparison_df[display_columns].sort_values(
            ["scenario_name", "policy_name"],
            ascending=True,
        ),
        use_container_width=True,
    )

    chart_columns = [
        "tokens_per_second",
        "avg_ttft_ms",
        "avg_latency_ms",
        "decode_batches_built",
        "kv_peak_used_blocks",
    ]

    available_chart_columns = [
        column for column in chart_columns if column in comparison_df.columns
    ]

    selected_metric = st.selectbox(
        "Comparison metric",
        options=available_chart_columns,
        index=0,
    )

    if "scenario_name" in comparison_df.columns and "policy_name" in comparison_df.columns:
        st.bar_chart(
            comparison_df,
            x="scenario_name",
            y=selected_metric,
            color="policy_name",
        )
def render_benchmark_explorer() -> None:
    st.header("Benchmark Explorer")

    st.markdown(
        """
        Explore benchmark CSVs produced by the runtime harness. This tab is meant to
        make throughput, backend latency, and KV cache pressure visible without
        digging through terminal output.
        """
    )

    benchmark_dir_raw = st.sidebar.text_input(
        "Benchmark directory",
        value=str(DEFAULT_BENCHMARK_DIR),
    )

    benchmark_dir = Path(benchmark_dir_raw)
    csv_files = list_benchmark_csvs(benchmark_dir)

    if not csv_files:
        st.warning(f"No benchmark CSV files found in `{benchmark_dir}`.")
        return

    csv_options = {
        str(path.relative_to(benchmark_dir)): path
        for path in csv_files
    }
    selected_csv_names = st.multiselect(
        "Benchmark CSVs",
        options=list(csv_options.keys()),
        default=list(csv_options.keys())[: min(12, len(csv_options))]
    )

    if not selected_csv_names:
        st.warning("Select at least one benchmark CSV.")
        return
    
    selected_csvs = [csv_options[name] for name in selected_csv_names]

    st.caption(f"Loaded `{len(selected_csvs)}` benchmark CSV file(s).")

    df = load_benchmark_csvs(
        selected_csvs,
        benchmark_dir=benchmark_dir
    )

    filtered_df = render_benchmark_filters(df)

    if filtered_df.empty:
        st.warning("No rows match the selected filters.")
        return

    render_benchmark_metric_cards(filtered_df)

    tab_policy, tab_concurrency, tab_block_size, tab_latency, tab_raw = st.tabs(
        [
            "Policy Comparison",
            "Concurrency Scaling",
            "Block Size Sensitivity",
            "Latency Table",
            "Raw Rows",
        ]
    )
    with tab_policy:
        render_policy_comparison_tab(filtered_df) 
    with tab_concurrency:
        render_concurrency_tab(filtered_df)

    with tab_block_size:
        render_block_size_tab(filtered_df)

    with tab_latency:
        render_latency_tab(filtered_df)

    with tab_raw:
        render_raw_data_tab(filtered_df)