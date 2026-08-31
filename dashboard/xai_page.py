"""Explainable AI dashboard page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from configs.config import OUTPUT_PATH
from xai.utils import load_xai_model, run_xai_pipeline

from .ui import info_tooltip, render_section_header


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
                explanations that highlight the regions contributing to the model's
                prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_tooltip(
        "xai_overview_info",
        "Explainable AI (XAI)",
        "Explainable AI techniques help researchers understand <em>which regions</em> "
        "of the input image contributed most to the model's prediction. This page uses "
        "Grad-CAM style visualizations to produce heatmaps and attention maps. "
        "XAI is intended to help researchers understand model behaviour — it does NOT "
        "prove the model's medical correctness.",
        "Deep learning models are often called 'black boxes' because even the engineers who build "
        "them don't always know exactly how they make their decisions. Explainable AI (XAI) is "
        "a set of tools designed to peer inside the box. When our AI looks at a scan and says 'I see a tumor here', "
        "XAI forces the AI to show its work by creating a 'heatmap'. The heatmap glows brightest over the exact "
        "pixels the AI was staring at when it made its decision. If the AI highlights the tumor, we can trust it. "
        "If it highlights the edge of the skull or some random background noise, we know the AI is "
        "making the right guess for the wrong reasons."
    )

    checkpoint_model = get_xai_model()
    if checkpoint_model is None:
        st.warning(
            "Trained checkpoint was not found. Please train the model before using the XAI page."
        )
        return

    render_section_header("Upload MRI", "Choose a medical image to generate predictions and explanations.")
    info_tooltip(
        "xai_upload_info",
        "Upload MRI",
        "Upload a medical image (MRI scan) to generate the predicted segmentation mask "
        "and explainability visualizations. The image will be preprocessed to match the "
        "model's input requirements.",
    )
    uploaded_file = st.file_uploader(
        "Upload MRI",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.info("Upload an MRI image to generate the explanation artifacts.")
        return

    run_clicked = st.button("Run Prediction", use_container_width=True)
    info_tooltip(
        "xai_run_info",
        "Run Prediction",
        "Click to run the U-Net model on the uploaded image and generate the "
        "segmentation mask, Grad-CAM heatmap, overlay, and attention visualization. "
        "This triggers a full forward pass through the model.",
        "This button sends the image through the AI's neural network to get the standard prediction, "
        "and then runs the complex XAI algorithms backwards through the network to generate the "
        "visual heatmaps."
    )
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

    info_tooltip(
        "xai_summary_info",
        "Prediction Summary",
        "<strong>Prediction Confidence</strong> — The maximum pixel-wise probability "
        "predicted by the model, indicating how confident the model is about the most "
        "likely foreground region.<br><br>"
        "<strong>Highlighted Region</strong> — Whether the model detected a region "
        "of interest above a confidence threshold.<br><br>"
        "<strong>Image Name</strong> — The uploaded filename for reference.",
    )

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
    info_tooltip(
        "xai_visuals_info",
        "XAI Visualizations",
        "<strong>Original MRI</strong> — The uploaded image in its original form.<br><br>"
        "<strong>Predicted Mask</strong> — The binary segmentation mask produced by "
        "the U-Net model.<br><br>"
        "<strong>Heatmap</strong> — A Grad-CAM style heatmap showing which spatial "
        "regions had the strongest influence on the model's prediction. Warmer colours "
        "indicate higher contribution.<br><br>"
        "<strong>Overlay</strong> — The heatmap superimposed on the original image for "
        "spatial reference.<br><br>"
        "<strong>Attention Visualization</strong> — An alternative visualization of the "
        "model's internal attention patterns.",
        "When analyzing these results, look closely at the <strong>Heatmap</strong> and <strong>Overlay</strong>. "
        "The red/orange areas are where the AI was focusing its 'eyes' the hardest. "
        "In a healthy, trustworthy AI, these glowing red hotspots should align perfectly with the "
        "white tumor shapes in the <strong>Predicted Mask</strong>. If they do not align, it means "
        "the AI is confused, even if its final prediction happened to be correct."
    )

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
