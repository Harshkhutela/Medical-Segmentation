"""Explainable AI dashboard page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from configs.config import OUTPUT_PATH
from xai.utils import load_xai_model, run_xai_pipeline

from .ui import render_section_header


@st.cache_resource(show_spinner=False)
def get_xai_model():
    """Load the trained model once for the XAI page."""
    try:
        return load_xai_model()
    except FileNotFoundError:
        return None


def _save_uploaded_image(uploaded_file) -> Path:
    """Persist the uploaded MRI to a temporary location for processing."""
    upload_dir = OUTPUT_PATH / "xai" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".png"
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=upload_dir,
    )
    temp_path = Path(temp_file.name)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_path


def render_xai_page() -> None:
    """Render the explainable AI page for MRI interpretation."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Explainable AI</div>
            <h1 style="margin:0 0 0.4rem 0;">Model Explanations for MRI Segmentation</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Upload an MRI, run the trained U-Net, and inspect Grad-CAM style
                explanations that highlight the tumor region.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checkpoint_model = get_xai_model()
    if checkpoint_model is None:
        st.warning(
            "Trained checkpoint was not found. Please train the model before using the XAI page."
        )
        return

    render_section_header("Upload MRI", "Choose a medical image to generate predictions and explanations.")
    uploaded_file = st.file_uploader(
        "Upload MRI",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.info("Upload an MRI image to generate the explanation artifacts.")
        return

    run_clicked = st.button("Run Prediction", use_container_width=True)
    if not run_clicked:
        st.info("Click Run Prediction to generate the mask, heatmap, and overlay.")
        return

    temp_path = _save_uploaded_image(uploaded_file)
    result = None
    try:
        with st.spinner("Generating XAI explanation..."):
            result = run_xai_pipeline(temp_path, model=checkpoint_model)
    except Exception as error:
        st.error(f"Unable to generate the XAI explanation: {error}")
        return
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    if result is None:
        st.error("XAI pipeline did not return any result.")
        return

    st.success("Prediction and explanation generated successfully.")

    render_section_header("Prediction Summary")
    summary_columns = st.columns(3)
    with summary_columns[0]:
        st.metric("Prediction Confidence", f"{result.confidence:.2%}")
    with summary_columns[1]:
        st.metric("Highlighted Region", "Detected" if "No confident" not in result.highlighted_region else "Not detected")
    with summary_columns[2]:
        st.metric("Image Name", uploaded_file.name)

    st.markdown(
        f"""
        <div class="dashboard-card">
            <strong>Explanation</strong>
            <p style="margin-top:0.6rem; color:#334155; line-height:1.75;">
                {result.explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("Segmentation and XAI Visuals")
    top_row = st.columns(3)
    with top_row[0]:
        st.image(result.original_image, caption="Original MRI", use_container_width=True)
    with top_row[1]:
        st.image(result.predicted_mask, caption="Predicted Mask", use_container_width=True)
    with top_row[2]:
        st.image(result.heatmap_image, caption="Heatmap", use_container_width=True)

    bottom_row = st.columns(2)
    with bottom_row[0]:
        st.image(result.overlay_image, caption="Overlay", use_container_width=True)
    with bottom_row[1]:
        st.image(result.attention_image, caption="Attention Visualization", use_container_width=True)

    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-top:1rem;">
            <strong>Saved XAI Artifacts</strong>
            <p style="margin-top:0.6rem; color:#334155; line-height:1.8;">
                Heatmap: {result.output_paths["heatmap"]}<br />
                Overlay: {result.output_paths["overlay"]}<br />
                Attention: {result.output_paths["attention"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
