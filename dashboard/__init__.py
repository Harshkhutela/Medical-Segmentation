"""Streamlit dashboard package for the medical segmentation project."""

from .benchmark_page import render_benchmark_page
from .inference_page import render_segmentation_page
from .ui import (
    render_about_page,
    render_active_learning_page,
    render_evaluation_page,
    render_footer,
    render_home_page,
    render_research_pipeline_page,
    render_sidebar_brand,
    render_training_results_page,
)

__all__ = [
    "render_about_page",
    "render_benchmark_page",
    "render_active_learning_page",
    "render_evaluation_page",
    "render_footer",
    "render_home_page",
    "render_research_pipeline_page",
    "render_sidebar_brand",
    "render_segmentation_page",
    "render_training_results_page",
]
