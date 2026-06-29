from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from experiments.dashboard.playground_views import render_runtime_playground
from experiments.dashboard.benchmark_views import render_benchmark_explorer
from experiments.dashboard.styles import inject_global_styles, render_hero


def render_kv_placeholder() -> None:
    st.header("KV Cache Inspector")
    st.info(
        "Next step: visualize KV blocks, request block tables, and block-size tradeoffs."
    )


def render_architecture_placeholder() -> None:
    st.header("Architecture Trace")

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
          └── decode batch construction
          ↓
        CustomLlamaDecodeEngine
          ├── batched TinyLlama decode
          ├── KVCachePool writes
          └── CUDA paged attention
          ↓
        Response + Metrics
        ```
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="TinyServe Lab",
        page_icon="⚡",
        layout="wide",
    )

    inject_global_styles()
    render_hero()

    tab_playground, tab_benchmarks, tab_kv, tab_architecture = st.tabs(
        [
            "Runtime Playground",
            "Benchmark Explorer",
            "KV Cache Inspector",
            "Architecture Trace",
        ]
    )

    with tab_playground:
        render_runtime_playground()

    with tab_benchmarks:
        render_benchmark_explorer()

    with tab_kv:
        render_kv_placeholder()

    with tab_architecture:
        render_architecture_placeholder()


if __name__ == "__main__":
    main()