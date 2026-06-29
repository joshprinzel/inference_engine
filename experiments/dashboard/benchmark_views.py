from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from experiments.dashboard.benchmark_data import (
    DEFAULT_BENCHMARK_DIR,
    aggregate_medians,
    filter_measured_rows,
    list_benchmark_csvs,
    load_benchmark_csv,
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

    cols = st.columns(6)

    if df.empty:
        for col in cols:
            col.metric("No data", "—")
        return

    correctness_rate = 0.0
    if "correctness_passed" in df.columns:
        correctness_rate = 100.0 * float(df["correctness_passed"].mean())

    cols[0].metric("Rows", str(len(df)))
    cols[1].metric("Median tok/s", f"{df['tokens_per_second'].median():.2f}")
    cols[2].metric("Median backend", f"{df['backend_ms_median'].median():.3f} ms")
    cols[3].metric("Median p95", f"{df['backend_ms_p95'].median():.3f} ms")
    cols[4].metric("Peak KV blocks", str(int(df["kv_peak_used_blocks"].max())))
    cols[5].metric("Correctness", f"{correctness_rate:.1f}%")


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
        "backend",
        "num_requests",
        "max_new_tokens",
        "block_size_tokens",
        "run_kind",
        *available,
    ]

    display_columns = [column for column in display_columns if column in df.columns]

    st.dataframe(
        df[display_columns].sort_values(
            ["num_requests", "max_new_tokens", "block_size_tokens"],
            ascending=True,
        ),
        use_container_width=True,
    )


def render_raw_data_tab(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True)


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

    csv_options = {path.name: path for path in csv_files}

    selected_csv_name = st.selectbox(
        "Benchmark CSV",
        options=list(csv_options.keys()),
    )

    selected_csv = csv_options[selected_csv_name]

    st.caption(f"Loaded `{selected_csv}`")

    df = load_benchmark_csv(selected_csv)
    filtered_df = render_benchmark_filters(df)

    if filtered_df.empty:
        st.warning("No rows match the selected filters.")
        return

    render_benchmark_metric_cards(filtered_df)

    tab_concurrency, tab_block_size, tab_latency, tab_raw = st.tabs(
        [
            "Concurrency Scaling",
            "Block Size Sensitivity",
            "Latency Table",
            "Raw Rows",
        ]
    )

    with tab_concurrency:
        render_concurrency_tab(filtered_df)

    with tab_block_size:
        render_block_size_tab(filtered_df)

    with tab_latency:
        render_latency_tab(filtered_df)

    with tab_raw:
        render_raw_data_tab(filtered_df)