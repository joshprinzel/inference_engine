from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from experiments.dashboard.playground_views import render_runtime_playground
from experiments.dashboard.benchmark_views import render_benchmark_explorer
from experiments.dashboard.architecture_views import render_architecture_trace
from experiments.dashboard.kv_views import render_kv_cache_inspector
from experiments.dashboard.styles import inject_global_styles, render_hero







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
        render_kv_cache_inspector()

    with tab_architecture:
        render_architecture_trace()


if __name__ == "__main__":
    main()