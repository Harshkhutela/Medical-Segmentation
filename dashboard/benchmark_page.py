"""Benchmark dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from benchmarking import BENCHMARK_PLOTS_DIR, load_benchmark_results, run_benchmark
from .ui import render_section_header


def _format_metric(value: float | int | None) -> str:
    """Format a benchmark value for display."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def render_benchmark_page() -> None:
    """Render the benchmark comparison page."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Benchmark</div>
            <h1 style="margin:0 0 0.4rem 0;">Model Comparison Framework</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Compare the trained U-Net against simple baselines to support
                research reporting and internship demonstrations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.7, 0.3], gap="large")
    with left:
        st.write(
            "This page compares the trained U-Net with a dummy baseline and a random predictor "
            "using Dice, IoU, Precision, Recall, and F1."
        )
    with right:
        run_button = st.button("Run Benchmark", use_container_width=True)

    frame = pd.DataFrame()
    if run_button:
        with st.spinner("Running benchmark..."):
            try:
                frame = run_benchmark()
                st.success("Benchmark completed successfully.")
            except Exception as error:
                st.error(f"Benchmark failed: {error}")
                return
    else:
        frame = load_benchmark_results()

    if frame.empty:
        st.info("No benchmark results found yet. Click Run Benchmark to generate them.")
        return

    render_section_header("Comparison Summary")
    summary_cards = st.columns(5)
    best_row = frame.sort_values("dice", ascending=False).iloc[0]
    with summary_cards[0]:
        st.metric("Best Dice", _format_metric(best_row.get("dice")))
    with summary_cards[1]:
        st.metric("Best IoU", _format_metric(best_row.get("iou")))
    with summary_cards[2]:
        st.metric("Best Precision", _format_metric(best_row.get("precision")))
    with summary_cards[3]:
        st.metric("Best Recall", _format_metric(best_row.get("recall")))
    with summary_cards[4]:
        st.metric("Best F1", _format_metric(best_row.get("f1")))

    render_section_header("Comparison Table")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    render_section_header("Benchmark Charts")
    for metric in ["dice", "iou", "precision", "recall", "f1"]:
        chart_path = BENCHMARK_PLOTS_DIR / f"{metric}_comparison.png"
        if chart_path.exists():
            st.image(str(chart_path), caption=metric.upper(), use_container_width=True)

    table_path = BENCHMARK_PLOTS_DIR / "benchmark_table.png"
    if table_path.exists():
        render_section_header("Rendered Comparison Table")
        st.image(str(table_path), use_container_width=True)
