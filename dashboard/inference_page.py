"""Segmentation inference page for the Streamlit dashboard."""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from configs.config import CHECKPOINT_PATH, DEVICE, OUTPUT_PATH
from configs.dataset_config import get_dataset_configuration
from inference import load_model
from utils.dataset import MedicalSegmentationDataset
from utils.metrics import dice_score, iou_score

from .ui import binary_mask_to_pil, encode_image, overlay_mask, render_section_header


@st.cache_resource(show_spinner=False)
def get_dashboard_model():
    """Load the trained checkpoint once and reuse it across reruns."""
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    if not checkpoint_file.is_file():
        return None

    try:
        return load_model(checkpoint_file)
    except (RuntimeError, OSError):
        return None


def _find_ground_truth_mask(upload_name: str) -> Path | None:
    """Find a matching mask for an uploaded file if one exists."""
    mask_directory = get_dataset_configuration().dataset_root / "masks"
    if not mask_directory.is_dir():
        return None

    upload_stem = Path(upload_name).stem
    for path in mask_directory.iterdir():
        if path.is_file() and path.stem == upload_stem:
            return path
    return None


def _preprocess_uploaded_image(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile):
    """Write the upload to a temp file and preprocess it with the dataset loader."""
    file_bytes = uploaded_file.getvalue()
    temp_directory = OUTPUT_PATH / "dashboard_uploads"
    temp_directory.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".png"
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=temp_directory,
    )
    temp_path = Path(temp_file.name)
    temp_file.write(file_bytes)
    temp_file.close()

    dataset_loader = MedicalSegmentationDataset()
    image_tensor = dataset_loader._load_image(temp_path)
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return temp_path, image, image_tensor


def _predict(model, image_tensor):
    """Run inference and return logits, probability mask, and binary mask."""
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        binary_mask = (probabilities >= 0.5).float()
    return logits, probabilities, binary_mask


def render_segmentation_page() -> None:
    """Render the segmentation inference interface."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Segmentation</div>
            <h1 style="margin:0 0 0.4rem 0;">Run Medical Image Inference</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Upload a medical image, run the trained U-Net, and compare the
                predicted mask against the original image in a polished
                research-demo layout.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    if not checkpoint_file.is_file():
        st.warning(
            "Trained checkpoint was not found. Run training before using the "
            "segmentation page."
        )
        return

    render_section_header("Upload Image", "Choose a medical image to inspect the model prediction.")
    uploaded_file = st.file_uploader(
        "Drop an image here or browse your files",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.info("Upload a medical image to start inference.")
        return

    model = get_dashboard_model()
    if model is None:
        st.error("Unable to load the saved checkpoint.")
        return

    temp_path, original_image, image_tensor = _preprocess_uploaded_image(
        uploaded_file
    )

    try:
        start_time = time.perf_counter()
        logits, probabilities, binary_mask = _predict(model, image_tensor)
        inference_time = time.perf_counter() - start_time
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    prediction_image = binary_mask_to_pil(binary_mask.cpu().numpy())
    overlay_image = overlay_mask(original_image, prediction_image)

    ground_truth_path = _find_ground_truth_mask(uploaded_file.name)
    dice_value = None
    iou_value = None
    if ground_truth_path is not None:
        dataset_loader = MedicalSegmentationDataset()
        ground_truth_mask = dataset_loader._load_mask(ground_truth_path)
        dice_value = dice_score(
            logits,
            ground_truth_mask.unsqueeze(0).to(DEVICE),
        )
        iou_value = iou_score(
            logits,
            ground_truth_mask.unsqueeze(0).to(DEVICE),
        )

    render_section_header("Prediction Results", "Original image, predicted mask, and overlay view.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(prediction_image, caption="Predicted Mask", use_container_width=True)
    with col3:
        st.image(overlay_image, caption="Overlay Image", use_container_width=True)

    metrics = st.columns(3)
    with metrics[0]:
        st.metric(
            "Dice Score",
            f"{dice_value:.4f}" if dice_value is not None else "N/A",
        )
    with metrics[1]:
        st.metric(
            "IoU Score",
            f"{iou_value:.4f}" if iou_value is not None else "N/A",
        )
    with metrics[2]:
        st.metric("Inference Time", f"{inference_time:.4f} s")

    if ground_truth_path is None:
        st.warning(
            "No matching ground truth mask was found for this uploaded image, "
            "so Dice and IoU could not be computed."
        )

    download_buffer = encode_image(prediction_image)
    st.download_button(
        label="Download Prediction",
        data=download_buffer,
        file_name=f"{Path(uploaded_file.name).stem}_prediction.png",
        mime="image/png",
    )
