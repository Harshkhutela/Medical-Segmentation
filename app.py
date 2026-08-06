"""Streamlit dashboard entry point for the medical segmentation project."""

from __future__ import annotations

import streamlit as st

from dashboard import (
    render_about_page,
    render_active_learning_page,
    render_benchmark_page,
    render_evaluation_page,
    render_footer,
    render_home_page,
    render_research_pipeline_page,
    render_segmentation_page,
    render_sidebar_brand,
    render_training_results_page,
)
from dashboard.ui import inject_global_styles
from dashboard.xai_page import render_xai_page


PAGE_OPTIONS = {
    "🏠 Home": render_home_page,
    "🧠 Segmentation": render_segmentation_page,
    "📊 Training Results": render_training_results_page,
    "🎯 Active Learning": render_active_learning_page,
    "🧪 Benchmark": render_benchmark_page,
    "Explainable AI": render_xai_page,
    "📈 Evaluation": render_evaluation_page,
    "🧭 Research Pipeline": render_research_pipeline_page,
    "ℹ About": render_about_page,
}


def main() -> None:
    """Configure the dashboard shell and route to the selected page."""
    st.set_page_config(
        page_title="Medical AI Dashboard",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_styles()

    with st.sidebar:
        render_sidebar_brand()
        st.markdown("---")
        page = st.radio("Navigate", list(PAGE_OPTIONS.keys()))

    PAGE_OPTIONS[page]()
    render_footer()


if __name__ == "__main__":
    main()
