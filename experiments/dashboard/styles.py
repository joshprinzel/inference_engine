from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(80, 120, 255, 0.12), rgba(120, 255, 200, 0.06));
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.85;
            line-height: 1.55;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid rgba(120, 120, 120, 0.25);
            background: rgba(80, 120, 255, 0.12);
        }

        .status-badge-good {
            background: rgba(30, 180, 120, 0.16);
        }

        .status-badge-warn {
            background: rgba(240, 170, 60, 0.18);
        }

        .section-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(120, 120, 120, 0.18);
            border-radius: 16px;
            margin-bottom: 1rem;
        }

        .small-muted {
            opacity: 0.72;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .code-label {
            font-size: 0.8rem;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">TinyServe Lab ⚡</div>
          <div class="hero-subtitle">
            A miniature LLM inference runtime with paged KV caching, CUDA paged attention,
            batched decode, and benchmark-driven systems analysis.
          </div>
          <div class="badge-row">
            <span class="status-badge status-badge-good">TinyLlama Runtime</span>
            <span class="status-badge status-badge-good">CUDA Paged Attention</span>
            <span class="status-badge status-badge-good">Physical KV Cache</span>
            <span class="status-badge status-badge-good">Batched Decode</span>
            <span class="status-badge">Benchmark Explorer</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badges(
    *,
    cuda_paged_attention: bool,
    kv_cache_pool: bool,
    batched_decode: bool,
    correctness: bool | None = None,
) -> None:
    correctness_label = "Correctness: unknown"
    correctness_class = ""

    if correctness is True:
        correctness_label = "Correctness: PASS"
        correctness_class = "status-badge-good"
    elif correctness is False:
        correctness_label = "Correctness: FAIL"
        correctness_class = "status-badge-warn"

    st.markdown(
        f"""
        <div class="badge-row">
            <span class="status-badge {'status-badge-good' if cuda_paged_attention else 'status-badge-warn'}">
                CUDA paged attention: {'ON' if cuda_paged_attention else 'OFF'}
            </span>
            <span class="status-badge {'status-badge-good' if kv_cache_pool else 'status-badge-warn'}">
                KV cache pool: {'ON' if kv_cache_pool else 'OFF'}
            </span>
            <span class="status-badge {'status-badge-good' if batched_decode else 'status-badge-warn'}">
                Batched decode: {'ON' if batched_decode else 'OFF'}
            </span>
            <span class="status-badge {correctness_class}">
                {correctness_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )