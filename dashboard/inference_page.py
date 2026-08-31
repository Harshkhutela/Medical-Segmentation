"""Segmentation inference page for the Streamlit dashboard."""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from configs.config import CHECKPOINT_PATH, DEVICE, IMAGE_SIZE, OUTPUT_PATH
from configs.dataset_config import get_dataset_configuration
from inference import load_model
from utils.dataset import MedicalSegmentationDataset
from utils.metrics import dice_score, iou_score, pixel_accuracy

from .ui import (
    binary_mask_to_pil,
    encode_image,
    info_tooltip,
    overlay_mask,
    render_section_header,
)

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _find_ground_truth_mask(upload_name: str) -> Path | None:
    """Find a matching mask for an uploaded file if one exists."""
    mask_directory = get_dataset_configuration().dataset_root / "masks"
    if not mask_directory.is_dir():
        return None

    upload_stem = Path(upload_name).stem

    # Direct stem match
    for path in mask_directory.iterdir():
        if path.is_file() and path.stem == upload_stem:
            return path

    # Normalized match (image_001 <-> mask_001)
    normalized_upload = _normalize_stem(upload_stem)
    for path in mask_directory.iterdir():
        if path.is_file() and _normalize_stem(path.stem) == normalized_upload:
            return path

    return None


def _normalize_stem(stem: str) -> str:
    """Normalize common image/mask naming patterns for matching."""
    normalized = stem.lower()
    for prefix in ("image_", "img_", "mask_", "msk_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    for suffix in ("_mask", "-mask", "_label", "-label", "_seg", "-seg"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


# ---------------------------------------------------------------------------
# Preprocessing & prediction
# ---------------------------------------------------------------------------

def _preprocess_uploaded_image(uploaded_file):
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
    """Run inference and return logits, probability tensor, and binary mask."""
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        binary_mask = (probabilities >= 0.5).float()
    return logits, probabilities, binary_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_extended_metrics(logits, ground_truth_mask):
    """Compute comprehensive segmentation metrics when ground truth exists."""
    gt_tensor = ground_truth_mask.unsqueeze(0).to(DEVICE)

    dice = dice_score(logits, gt_tensor)
    iou = iou_score(logits, gt_tensor)
    pix_acc = pixel_accuracy(logits, gt_tensor)

    # Compute precision, recall, F1 from confusion counts
    probs = torch.sigmoid(logits)
    pred_binary = (probs >= 0.5).float()
    gt_binary = (gt_tensor > 0).float()
    if gt_binary.dim() == 3:
        gt_binary = gt_binary.unsqueeze(1)

    tp = float((pred_binary * gt_binary).sum().item())
    fp = float((pred_binary * (1 - gt_binary)).sum().item())
    fn = float(((1 - pred_binary) * gt_binary).sum().item())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": pix_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _compute_stats_at_threshold(prob_np: np.ndarray, threshold: float):
    """Compute foreground statistics from a probability array at a threshold."""
    binary = (prob_np >= threshold).astype(np.float32)
    total_pixels = binary.size
    fg_pixels = int(binary.sum())
    coverage_pct = (fg_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    max_prob = float(prob_np.max())
    mean_prob = float(prob_np.mean())

    return {
        "foreground_pixels": fg_pixels,
        "total_pixels": total_pixels,
        "coverage_pct": coverage_pct,
        "max_probability": max_prob,
        "mean_probability": mean_prob,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Probability heatmap
# ---------------------------------------------------------------------------

def _create_probability_heatmap(prob_np: np.ndarray) -> Image.Image:
    """Convert a 2-D probability array into a jet-coloured heatmap PIL image."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=120)
    im = ax.imshow(prob_np, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tumor Probability", fontsize=8, color="#334155")
    cbar.ax.tick_params(labelsize=7, colors="#64748b")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------

_SS_PREFIX = "seg_"
_SS_UPLOAD_ID = f"{_SS_PREFIX}upload_id"
_SS_ORIGINAL_IMAGE = f"{_SS_PREFIX}orig_img"
_SS_PROB_NP = f"{_SS_PREFIX}prob_np"
_SS_LOGITS = f"{_SS_PREFIX}logits"
_SS_INFERENCE_TIME = f"{_SS_PREFIX}inf_time"
_SS_HEATMAP = f"{_SS_PREFIX}heatmap"
_SS_UPLOAD_NAME = f"{_SS_PREFIX}upload_name"
_SS_IMG_DIMS = f"{_SS_PREFIX}img_dims"


def _run_or_cache_inference(uploaded_file, model):
    """Run inference only when the uploaded file changes, otherwise return cache."""
    upload_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get(_SS_UPLOAD_ID) == upload_id:
        return (
            st.session_state[_SS_ORIGINAL_IMAGE],
            st.session_state[_SS_PROB_NP],
            st.session_state[_SS_LOGITS],
            st.session_state[_SS_INFERENCE_TIME],
            st.session_state[_SS_HEATMAP],
            st.session_state[_SS_IMG_DIMS],
        )

    # New upload — run inference
    temp_path, original_image, image_tensor = _preprocess_uploaded_image(uploaded_file)

    try:
        start_time = time.perf_counter()
        logits, probabilities, _ = _predict(model, image_tensor)
        inference_time = time.perf_counter() - start_time
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    prob_np = probabilities.squeeze().cpu().numpy()
    heatmap_image = _create_probability_heatmap(prob_np)

    file_bytes = uploaded_file.getvalue()
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_dims = pil_img.size  # (width, height)

    # Cache everything
    st.session_state[_SS_UPLOAD_ID] = upload_id
    st.session_state[_SS_ORIGINAL_IMAGE] = original_image
    st.session_state[_SS_PROB_NP] = prob_np
    st.session_state[_SS_LOGITS] = logits
    st.session_state[_SS_INFERENCE_TIME] = inference_time
    st.session_state[_SS_HEATMAP] = heatmap_image
    st.session_state[_SS_UPLOAD_NAME] = uploaded_file.name
    st.session_state[_SS_IMG_DIMS] = img_dims

    return original_image, prob_np, logits, inference_time, heatmap_image, img_dims


def _run_or_cache_inference_from_path(image_path: Path, model):
    """Run inference on a disk-based image path, with session-state caching."""
    upload_id = f"sample_{image_path.name}"

    if st.session_state.get(_SS_UPLOAD_ID) == upload_id:
        return (
            st.session_state[_SS_ORIGINAL_IMAGE],
            st.session_state[_SS_PROB_NP],
            st.session_state[_SS_LOGITS],
            st.session_state[_SS_INFERENCE_TIME],
            st.session_state[_SS_HEATMAP],
            st.session_state[_SS_IMG_DIMS],
        )

    dataset_loader = MedicalSegmentationDataset()
    image_tensor = dataset_loader._load_image(image_path)
    original_image = Image.open(str(image_path)).convert("RGB")
    img_dims = original_image.size

    start_time = time.perf_counter()
    logits, probabilities, _ = _predict(model, image_tensor)
    inference_time = time.perf_counter() - start_time

    prob_np = probabilities.squeeze().cpu().numpy()
    heatmap_image = _create_probability_heatmap(prob_np)

    st.session_state[_SS_UPLOAD_ID] = upload_id
    st.session_state[_SS_ORIGINAL_IMAGE] = original_image
    st.session_state[_SS_PROB_NP] = prob_np
    st.session_state[_SS_LOGITS] = logits
    st.session_state[_SS_INFERENCE_TIME] = inference_time
    st.session_state[_SS_HEATMAP] = heatmap_image
    st.session_state[_SS_UPLOAD_NAME] = image_path.name
    st.session_state[_SS_IMG_DIMS] = img_dims

    return original_image, prob_np, logits, inference_time, heatmap_image, img_dims


@st.cache_data(show_spinner=False)
def _get_sample_image_list():
    """Scan the dataset directory and return a list of sample images with labels."""
    dataset_config = get_dataset_configuration()
    images_dir = dataset_config.dataset_root / "images"
    masks_dir = dataset_config.dataset_root / "masks"

    if not images_dir.is_dir():
        return []

    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue

        # Check if corresponding mask has content
        has_target = False
        mask_path = masks_dir / img_path.name
        if mask_path.exists():
            try:
                mask_img = Image.open(str(mask_path)).convert("L")
                mask_np = np.array(mask_img)
                has_target = mask_np.max() > 0
            except Exception:
                pass

        samples.append((img_path, has_target))
        if len(samples) >= 30:  # Limit to first 30 for performance
            break

    return samples


def _get_sample_image_selector():
    """Render a selectbox for choosing a sample dataset image."""
    sample_list = _get_sample_image_list()
    if not sample_list:
        st.caption("No dataset images found.")
        return None

    # Sort: images WITH targets first
    with_target = [(p, t) for p, t in sample_list if t]
    without_target = [(p, t) for p, t in sample_list if not t]
    sorted_samples = with_target + without_target

    options = ["— Select a sample image —"] + [
        f"{p.name}  {'✅ (has target)' if t else '⬜ (no target)'}"
        for p, t in sorted_samples
    ]

    choice = st.selectbox("Sample Image", options, index=0, label_visibility="collapsed")
    if choice == "— Select a sample image —":
        return None

    idx = options.index(choice) - 1
    return sorted_samples[idx][0]


# ===================================================================
# Main page renderer
# ===================================================================

def render_segmentation_page() -> None:
    """Render the segmentation inference interface."""
    device_name = "NVIDIA GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    gpu_name = ""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA GPU"
    device_label = gpu_name if gpu_name else device_name

    # --- Hero ---
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">Medical Image Segmentation</div>
            <h1 style="margin:0 0 0.4rem 0; font-size:1.9rem; font-weight:800; color:#0f172a;">
                Medical Image Segmentation
            </h1>
            <p style="margin:0 0 1rem 0; color:#475569; line-height:1.7; max-width:860px;">
                U-Net based pixel-level segmentation with confidence-aware prediction analysis.
                Upload a medical image to run the trained model and inspect the predicted
                segmentation mask, probability heatmap, and comprehensive metrics.
            </p>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                <span style="display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.75rem;border-radius:999px;background:#f0fdf4;border:1px solid #bbf7d0;color:#15803d;font-size:0.75rem;font-weight:600;">
                    ● {device_label}
                </span>
                <span style="display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.75rem;border-radius:999px;background:#f0f9ff;border:1px solid #bae6fd;color:#0369a1;font-size:0.75rem;font-weight:600;">
                    U-Net · {IMAGE_SIZE}×{IMAGE_SIZE}
                </span>
                <span style="display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.75rem;border-radius:999px;background:#faf5ff;border:1px solid #e9d5ff;color:#7c3aed;font-size:0.75rem;font-weight:600;">
                    best_model.pth
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Checkpoint check ---
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    if not checkpoint_file.is_file():
        st.warning(
            "Trained checkpoint was not found. Run training before using the "
            "segmentation page."
        )
        return

    # --- Upload Section ---
    render_section_header(
        "Upload Image",
        "Select a medical image to run segmentation inference.",
    )
    info_tooltip(
        "upload_info",
        "Image Upload",
        "Upload a medical image (PNG, JPG, BMP, TIF) to run the trained U-Net model. "
        "The image will be resized to the model's input resolution "
        f"({IMAGE_SIZE}×{IMAGE_SIZE}) and normalized before inference. "
        "The original pixel values are preserved for display.",
    )

    uploaded_file = st.file_uploader(
        "Drop an image here or browse your files",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    # --- Sample Image Loader ---
    render_section_header(
        "Or Load a Sample Image",
        "Pick an image from the dataset to test the model instantly.",
    )
    info_tooltip(
        "sample_info",
        "Sample Images",
        "These are actual images from the project dataset. Images marked "
        "'(has target)' contain ground truth tumor regions — the model should "
        "produce a visible segmentation mask for these. Images marked '(no target)' "
        "are healthy scans — the model should correctly predict an empty mask.",
    )

    _sample_image_path = _get_sample_image_selector()

    # Determine which source to use
    use_sample = _sample_image_path is not None and uploaded_file is None
    use_upload = uploaded_file is not None

    if not use_sample and not use_upload:
        st.info("Upload a medical image or select a sample image to start inference.")
        return

    # --- Load model ---
    model = get_dashboard_model()
    if model is None:
        st.error("Unable to load the saved checkpoint.")
        return

    # --- Run or retrieve cached inference ---
    if use_upload:
        # Show upload info
        file_ext = Path(uploaded_file.name).suffix.upper().lstrip(".")
        st.markdown(
            f"""
            <div class="dashboard-card" style="margin:0.8rem 0;">
                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:center;">
                    <div><span style="color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">Filename</span>
                    <div style="font-weight:700;color:#0f172a;margin-top:0.15rem;">{uploaded_file.name}</div></div>
                    <div><span style="color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">File Type</span>
                    <div style="font-weight:700;color:#0f172a;margin-top:0.15rem;">{file_ext}</div></div>
                    <div><span style="color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">File Size</span>
                    <div style="font-weight:700;color:#0f172a;margin-top:0.15rem;">{uploaded_file.size / 1024:.1f} KB</div></div>
                    <div><span style="color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">Model Input</span>
                    <div style="font-weight:700;color:#0f172a;margin-top:0.15rem;">{IMAGE_SIZE}×{IMAGE_SIZE}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        (
            original_image,
            prob_np,
            logits,
            inference_time,
            heatmap_image,
            img_dims,
        ) = _run_or_cache_inference(uploaded_file, model)
        image_name = uploaded_file.name
    else:
        st.markdown(
            f"""
            <div class="dashboard-card" style="margin:0.8rem 0; border-left:4px solid #0ea5e9;">
                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:center;">
                    <div><span style="color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">Sample Loaded</span>
                    <div style="font-weight:700;color:#0f172a;margin-top:0.15rem;">{_sample_image_path.name}</div></div>
                    <div><span style="color:#64748b;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">Source</span>
                    <div style="font-weight:700;color:#0f172a;margin-top:0.15rem;">Dataset</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        (
            original_image,
            prob_np,
            logits,
            inference_time,
            heatmap_image,
            img_dims,
        ) = _run_or_cache_inference_from_path(_sample_image_path, model)
        image_name = _sample_image_path.name

    # --- Out of Domain Warning ---
    if prob_np.max() < 0.005 and use_upload:
        st.markdown(
            """
            <div class="dashboard-card" style="border-left:4px solid #ef4444; margin:1rem 0; background:#fef2f2;">
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                    <span style="font-size:1.2rem;">⚠️</span>
                    <span style="font-size:1.05rem;font-weight:800;color:#991b1b;">
                        This is not a valid scan or a medical image
                    </span>
                </div>
                <p style="margin:0 0 0.8rem 0;color:#7f1d1d;font-size:0.9rem;line-height:1.7;">
                    The model predicted an extremely low probability across the entire image. 
                    If you uploaded a screenshot, a non-medical image, or an image format that 
                    the model was not trained on, the prediction will be near zero. 
                    <strong>Try using the "Load Sample Image" dropdown above to test with real dataset images.</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Threshold control ---
    render_section_header(
        "Segmentation Threshold",
        "Adjust the binary classification threshold for the predicted mask.",
    )
    info_tooltip(
        "threshold_info",
        "Segmentation Threshold",
        "The model produces continuous probabilities (0 to 1) for each pixel. "
        "Pixels with probability above this threshold are classified as foreground "
        "(target region) in the binary segmentation mask. Lowering the threshold "
        "increases sensitivity (more foreground pixels) but may also increase false "
        "positives. The default medical standard threshold is 0.50.",
        "Think of the threshold like a strict security guard at a door. The AI gives every "
        "pixel a 'suspicion score' from 0% to 100%. If the security guard's strictness (threshold) "
        "is set to 50%, only pixels that are more than 50% suspicious are allowed to be labeled "
        "as a tumor. If you lower it to 5%, the guard becomes very paranoid and lets almost anything "
        "through, creating a massive (but likely inaccurate) tumor boundary."
    )

    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        format="%.2f",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div style="font-size:0.85rem;color:#475569;margin-top:-0.5rem;">Current threshold: <strong>{threshold:.2f}</strong></div>',
        unsafe_allow_html=True,
    )

    # Compute binary mask at selected threshold (no re-inference)
    binary_np = (prob_np >= threshold).astype(np.float32)
    prediction_image = binary_mask_to_pil(binary_np[np.newaxis, ...])
    stats = _compute_stats_at_threshold(prob_np, threshold)

    # --- Prediction Results ---
    render_section_header(
        "Prediction Results",
        "Original image, binary mask, probability heatmap, and overlay.",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="dashboard-card" style="text-align:center;">'
            '<div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">Original Image</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(original_image, use_container_width=True)
        info_tooltip(
            "orig_info",
            "Original Image",
            "The uploaded medical image displayed in its original RGB form. "
            "Before inference, the image is resized and normalized to match the "
            "training preprocessing pipeline.",
            "This is just the raw image exactly as you provided it, before the AI does any math."
        )

    with col2:
        st.markdown(
            '<div class="dashboard-card" style="text-align:center;">'
            f'<div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">Binary Mask (t={threshold:.2f})</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(prediction_image, use_container_width=True)
        info_tooltip(
            "mask_info",
            "Binary Predicted Mask",
            "The binary segmentation mask produced by thresholding the model's "
            "probability output. White pixels indicate regions where the model's "
            f"predicted probability exceeds the current threshold ({threshold:.2f}). "
            "Black pixels indicate regions below the threshold.",
            "This is the final, concrete answer from the AI. Anywhere you see white is where "
            "the AI is firmly stating 'I believe a tumor is here'. Everything black is considered "
            "healthy tissue."
        )

    with col3:
        st.markdown(
            '<div class="dashboard-card" style="text-align:center;">'
            '<div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">Probability Heatmap</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(heatmap_image, use_container_width=True)
        info_tooltip(
            "heatmap_info",
            "Probability Heatmap",
            "Each pixel represents the model's estimated probability of belonging to "
            "the target segmentation class, displayed on a jet colour scale from 0 (blue) "
            "to 1 (red). This visualization shows the raw model output before any "
            "thresholding is applied. For images with very low probabilities, the heatmap "
            "may appear almost uniformly blue, which is expected behaviour.",
            "This is the AI's 'suspicion map'. Instead of a strict yes/no like the binary mask, "
            "this shows exactly how worried the AI is about every single pixel. Dark blue means 0% worry. "
            "Bright red means 100% worry. This map is incredibly useful because if the AI completely misses "
            "a tumor in the final white mask, you can look here to see if it was at least <i>slightly</i> "
            "suspicious (light blue/green) or if it completely ignored it."
        )

    with col4:
        overlay_image = overlay_mask(original_image, prediction_image)
        st.markdown(
            '<div class="dashboard-card" style="text-align:center;">'
            '<div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">Overlay</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.image(overlay_image, use_container_width=True)
        info_tooltip(
            "overlay_info",
            "Overlay Visualization",
            "The original image with the binary predicted mask overlaid in a "
            "semi-transparent red colour. This helps visually assess whether the "
            "predicted segmentation aligns with the visible anatomy.",
            "This simply takes the AI's final white mask and slaps it on top of the original image "
            "as a red blob. It's the fastest way for a doctor to see exactly what part of the brain "
            "the AI is pointing at."
        )

    # --- Overlay controls ---
    render_section_header("Overlay Controls")
    info_tooltip(
        "overlay_ctrl_info",
        "Overlay Opacity",
        "Adjust the transparency of the red mask overlay on the original image. "
        "Higher values make the mask more opaque; lower values make it more transparent. "
        "This control does NOT re-run model inference.",
        "Just a slider to make the red overlay blob more see-through, so you can look at the "
        "actual brain tissue underneath it."
    )
    opacity_slider = st.slider(
        "Mask Overlay Opacity",
        min_value=0,
        max_value=255,
        value=120,
        step=5,
        label_visibility="collapsed",
    )
    if opacity_slider != 120:
        adjusted_overlay = overlay_mask(
            original_image, prediction_image, alpha=opacity_slider
        )
        st.image(adjusted_overlay, caption="Adjusted Overlay", use_container_width=True)

    # --- Prediction Metrics ---
    render_section_header(
        "Prediction Metrics",
        "Quantitative analysis of the model prediction.",
    )

    row1 = st.columns(4)
    with row1[0]:
        st.metric("Max Probability", f"{stats['max_probability']:.6f}")
    with row1[1]:
        st.metric("Mean Probability", f"{stats['mean_probability']:.6f}")
    with row1[2]:
        st.metric("Foreground Pixels", f"{stats['foreground_pixels']:,}")
    with row1[3]:
        st.metric("Mask Coverage", f"{stats['coverage_pct']:.2f}%")

    row2 = st.columns(4)
    with row2[0]:
        st.metric("Inference Time", f"{inference_time:.4f} s")
    with row2[1]:
        st.metric("Input Resolution", f"{IMAGE_SIZE}×{IMAGE_SIZE}")
    with row2[2]:
        st.metric("Device", device_label)
    with row2[3]:
        st.metric("Threshold", f"{threshold:.2f}")

    # Metric explanations
    info_tooltip(
        "metrics_info",
        "Prediction Metrics Explained",
        "<strong>Max Probability</strong> — The highest pixel-wise probability "
        "predicted by the model across the entire image. A value close to 1.0 "
        "indicates high confidence that at least one pixel belongs to the target class.<br><br>"
        "<strong>Mean Probability</strong> — The average predicted probability across "
        "all pixels. Low values indicate the model expects most of the image to be background.<br><br>"
        "<strong>Foreground Pixels</strong> — Number of pixels whose predicted probability "
        f"exceeds the current threshold ({threshold:.2f}).<br><br>"
        "<strong>Mask Coverage</strong> — Foreground pixels divided by total pixels, "
        "expressed as a percentage.<br><br>"
        "<strong>Inference Time</strong> — Wall-clock time for the model's forward pass. "
        "This is measured once and does not change when you adjust the threshold or overlay.<br><br>"
        "<strong>Input Resolution</strong> — The spatial dimensions to which the uploaded "
        f"image is resized before being passed to the model ({IMAGE_SIZE}×{IMAGE_SIZE}).<br><br>"
        "<strong>Device</strong> — The hardware used for inference (CPU or GPU).<br><br>"
        "<strong>Threshold</strong> — The current binary segmentation threshold.",
        "These metrics give you the exact math behind the AI's guess. "
        "<strong>Max Probability</strong> is the AI's 'peak worry'. If it's 0.99, the AI is "
        "absolutely terrified of at least one pixel. <strong>Mean Probability</strong> is the 'average worry' "
        "across the whole image (which is usually very low since most of a scan is just empty space or healthy brain). "
        "<strong>Foreground Pixels</strong> is literally just counting how many pixels turned white in the final mask."
    )

    # --- Zero-foreground handling ---
    if stats["foreground_pixels"] == 0:
        if prob_np.max() >= 0.005:  # It's a valid medical image, just healthy
            st.markdown(
                f"""
                <div class="dashboard-card" style="border-left:4px solid #22c55e; margin:1rem 0; background:#f0fdf4;">
                    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                        <span style="font-size:1.2rem;">✅</span>
                        <span style="font-size:1.05rem;font-weight:800;color:#166534;">
                            Healthy Scan: No tumor detected
                        </span>
                    </div>
                    <p style="margin:0 0 0.8rem 0;color:#14532d;font-size:0.9rem;line-height:1.7;">
                        There are highest chances that this person is healthy, no chances of tumor detected 
                        above the current segmentation threshold (<strong>{threshold:.2f}</strong>). The predicted mask will remain entirely black.
                    </p>
                    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                        <div>
                            <span style="color:#166534;font-size:0.72rem;font-weight:700;text-transform:uppercase;">Max Predicted Probability</span>
                            <div style="font-weight:800;color:#14532d;font-size:1.1rem;">{stats['max_probability']:.6f}</div>
                        </div>
                        <div>
                            <span style="color:#166534;font-size:0.72rem;font-weight:700;text-transform:uppercase;">Current Threshold</span>
                            <div style="font-weight:800;color:#14532d;font-size:1.1rem;">{threshold:.2f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else: # Extremely low probability handled by out-of-domain warning, but we still show the metrics
            st.markdown(
                f"""
                <div class="dashboard-card" style="border-left:4px solid #f59e0b; margin:1rem 0;">
                    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                        <span style="font-size:1.2rem;">⚠️</span>
                        <span style="font-size:1.05rem;font-weight:800;color:#0f172a;">
                            No Foreground Region Detected
                        </span>
                    </div>
                    <p style="margin:0 0 0.8rem 0;color:#475569;font-size:0.9rem;line-height:1.7;">
                        No foreground region was detected.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Explore lower thresholds
        render_section_header("Explore Lower Thresholds")
        info_tooltip(
            "explore_thresh_info",
            "Threshold Exploration",
            "This section shows what the binary mask would look like at progressively "
            "lower thresholds. Lowering the threshold does NOT make the prediction more "
            "accurate — it is an exploratory visualization to inspect whether the model "
            "assigns any non-zero probability to specific regions. Use this to understand "
            "the model's confidence distribution across the image.",
        )

        explore_thresholds = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
        cols = st.columns(len(explore_thresholds))
        for col, thr in zip(cols, explore_thresholds):
            with col:
                thr_binary = (prob_np >= thr).astype(np.float32)
                thr_fg = int(thr_binary.sum())
                thr_img = binary_mask_to_pil(thr_binary[np.newaxis, ...])
                st.image(thr_img, caption=f"t={thr:.2f}", use_container_width=True)
                st.markdown(
                    f'<div style="text-align:center;font-size:0.78rem;color:#475569;">FG: {thr_fg:,}</div>',
                    unsafe_allow_html=True,
                )

    # --- Ground Truth Check & Metrics ---
    ground_truth_path = _find_ground_truth_mask(image_name)

    if ground_truth_path is not None:
        dataset_loader = MedicalSegmentationDataset()
        ground_truth_mask = dataset_loader._load_mask(ground_truth_path)
        metrics = _compute_extended_metrics(logits, ground_truth_mask)

        render_section_header(
            "Segmentation Metrics — Ground Truth Available",
            "Comprehensive evaluation against the matched ground truth mask.",
        )

        m_row1 = st.columns(4)
        with m_row1[0]:
            st.metric("Dice Score", f"{metrics['dice']:.4f}")
        with m_row1[1]:
            st.metric("IoU Score", f"{metrics['iou']:.4f}")
        with m_row1[2]:
            st.metric("Pixel Accuracy", f"{metrics['pixel_accuracy']:.4f}")
        with m_row1[3]:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")

        m_row2 = st.columns(4)
        with m_row2[0]:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with m_row2[1]:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with m_row2[2]:
            st.metric("Inference Time", f"{inference_time:.4f} s")
        with m_row2[3]:
            st.metric("Device", device_label)

        info_tooltip(
            "gt_metrics_info",
            "Evaluation Metrics Explained",
            "<strong>Dice Score</strong> — Measures overlap between prediction and ground truth. "
            "Dice = 2|Prediction ∩ GT| / (|Prediction| + |GT|). Range: 0 (no overlap) to 1 (perfect). "
            "This is an <em>evaluation metric</em>, not a measure of model confidence.<br><br>"
            "<strong>IoU (Intersection over Union)</strong> — Measures overlap as "
            "IoU = |Prediction ∩ GT| / |Prediction ∪ GT|. Range: 0 to 1. Higher is better.<br><br>"
            "<strong>Pixel Accuracy</strong> — Fraction of correctly classified pixels.<br><br>"
            "<strong>Precision</strong> — Of all pixels predicted as foreground, what fraction "
            "actually belongs to the target class in the ground truth.<br><br>"
            "<strong>Recall</strong> — Of all actual foreground pixels in the ground truth, "
            "what fraction was correctly detected by the model.<br><br>"
            "<strong>F1 Score</strong> — Harmonic mean of precision and recall.",
        )

        # Ground truth comparison
        render_section_header("Ground Truth Comparison")
        gt_col1, gt_col2 = st.columns(2)
        with gt_col1:
            gt_image = Image.open(str(ground_truth_path)).convert("L")
            st.image(gt_image, caption="Ground Truth Mask", use_container_width=True)
        with gt_col2:
            st.image(
                prediction_image, caption="Predicted Mask", use_container_width=True
            )

    else:
        # No ground truth
        st.markdown(
            """
            <div class="dashboard-card" style="border-left:4px solid #0d9488; margin:1rem 0;">
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
                    <span style="font-size:1.2rem;">ℹ️</span>
                    <span style="font-size:1.05rem;font-weight:800;color:#0f172a;">
                        No Ground Truth Mask Available
                    </span>
                </div>
                <p style="margin:0;color:#475569;font-size:0.88rem;line-height:1.7;">
                    No matching ground truth mask was found for this image. Dice and IoU
                    metrics require a ground truth segmentation mask for comparison. Since
                    no ground truth is available, these evaluation metrics cannot be computed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        no_gt_row = st.columns(4)
        with no_gt_row[0]:
            st.metric("Dice Score", "N/A")
        with no_gt_row[1]:
            st.metric("IoU Score", "N/A")
        with no_gt_row[2]:
            st.metric("Pixel Accuracy", "N/A")
        with no_gt_row[3]:
            st.metric("F1 Score", "N/A")

        info_tooltip(
            "no_gt_info",
            "Why are Dice and IoU N/A?",
            "Dice and IoU are <em>evaluation metrics</em> that measure the overlap between "
            "a predicted mask and a ground truth mask. They require a labelled reference mask "
            "for comparison. Since no ground truth mask was found for this uploaded image, "
            "these metrics cannot be computed. This does NOT mean the prediction is wrong — "
            "it simply means there is no reference to evaluate against.",
        )

    # --- Downloads ---
    render_section_header("Download Results")
    info_tooltip(
        "download_info",
        "Download",
        "Download the predicted mask, overlay visualization, probability heatmap, "
        "or original image as PNG files for offline analysis or reporting.",
    )
    dl_cols = st.columns(4)

    with dl_cols[0]:
        mask_buffer = encode_image(prediction_image)
        st.download_button(
            label="Predicted Mask",
            data=mask_buffer,
            file_name=f"{Path(image_name).stem}_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl_cols[1]:
        overlay_buffer = encode_image(overlay_image)
        st.download_button(
            label="Overlay",
            data=overlay_buffer,
            file_name=f"{Path(image_name).stem}_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl_cols[2]:
        heatmap_buffer = encode_image(heatmap_image)
        st.download_button(
            label="Probability Heatmap",
            data=heatmap_buffer,
            file_name=f"{Path(image_name).stem}_heatmap.png",
            mime="image/png",
            use_container_width=True,
        )

    with dl_cols[3]:
        orig_buffer = encode_image(original_image)
        st.download_button(
            label="Original Image",
            data=orig_buffer,
            file_name=f"{Path(image_name).stem}_original.png",
            mime="image/png",
            use_container_width=True,
        )
