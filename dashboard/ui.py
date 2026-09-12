"""Shared Streamlit UI helpers for the medical segmentation dashboard."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from configs.config import BASE_DIR, OUTPUT_PATH
from configs.dataset_config import get_dataset_configuration


APP_TITLE = "Medical AI Dashboard"
PROJECT_TITLE = "Semi-Supervised Active Learning for Medical Image Segmentation"
APP_VERSION = "v2.0.0"
DEVELOPER_NAME = "Harsh Khutela"
DEVELOPER_ROLE = "Engineer and Research Intern at CSIR 4PI"
GITHUB_PLACEHOLDER = "https://github.com/your-repo"
LICENSE_NAME = "MIT License"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HISTORY_PATH = OUTPUT_PATH / "history.csv"
LARGE_HISTORY_PATH = OUTPUT_PATH / "large_history.csv"
EXPERIMENT_RESULTS_PATH = OUTPUT_PATH / "experiment_results.csv"
FINAL_METRICS_PATH = OUTPUT_PATH / "final_metrics.csv"
ACTIVE_LEARNING_PATH = OUTPUT_PATH / "active_learning_scores.csv"
SEMI_SUPERVISED_REPORT_PATH = OUTPUT_PATH / "semi_supervised_report.txt"
PLOTS_DIR = OUTPUT_PATH / "plots"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PREDICTION_DIRS = [OUTPUT_PATH / "epoch_predictions", OUTPUT_PATH / "predictions"]


def inject_global_styles() -> None:
    """Apply a polished medical research theme to the entire dashboard."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --font-primary: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                --bg-main: #ffffff;
                --bg-subtle: #f8fafc;
                --card-border: #e2e8f0;
                --accent-teal: #0d9488;
                --accent-teal-dark: #0f766e;
                --accent-blue: #0284c7;
                --text-main: #0f172a;
                --text-muted: #64748b;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes pulseGlow {
                0% { box-shadow: 0 0 0 0 rgba(13, 148, 136, 0.4); }
                70% { box-shadow: 0 0 0 8px rgba(13, 148, 136, 0); }
                100% { box-shadow: 0 0 0 0 rgba(13, 148, 136, 0); }
            }

            html, body, [data-testid="stAppViewContainer"] {
                font-family: var(--font-primary) !important;
                background: #f8fafc !important;
                color: var(--text-main) !important;
                -webkit-font-smoothing: antialiased;
            }

            .stApp {
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
                animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            }

            .block-container {
                padding-top: 1.6rem !important;
                padding-bottom: 2.5rem !important;
                max-width: 1380px !important;
            }

            /* Pure Crisp Light Sidebar Styling */
            [data-testid="stSidebar"] {
                background: #ffffff !important;
                border-right: 1px solid #e2e8f0 !important;
                box-shadow: 4px 0 20px rgba(0, 0, 0, 0.02) !important;
            }

            [data-testid="stSidebar"] * {
                color: #0f172a !important;
            }

            .sidebar-profile-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.1rem;
                margin-top: 1.2rem;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
                transition: all 0.25s ease;
            }

            .sidebar-profile-card:hover {
                border-color: #cbd5e1;
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
            }

            .sidebar-profile-avatar {
                width: 44px;
                height: 44px;
                border-radius: 12px;
                background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                font-size: 1.1rem;
                color: #ffffff !important;
                box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3);
                letter-spacing: -0.02em;
            }

            .sidebar-profile-name {
                font-size: 1.02rem;
                font-weight: 700;
                color: #0f172a !important;
                letter-spacing: -0.01em;
            }

            .sidebar-profile-role {
                font-size: 0.78rem;
                color: #64748b !important;
                font-weight: 500;
            }

            .sidebar-status-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                border-radius: 999px;
                background: #f0fdf4;
                border: 1px solid #bbf7d0;
                color: #15803d !important;
                font-size: 0.72rem;
                font-weight: 600;
                margin-top: 0.65rem;
            }

            .status-dot {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background-color: #16a34a;
                box-shadow: 0 0 6px #16a34a;
                animation: pulseGlow 2s infinite;
            }

            /* Sidebar Radio Navigation Override (Clean White Light Theme) */
            [data-testid="stSidebar"] [data-testid="stRadio"] > label {
                font-size: 0.75rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.08em !important;
                color: #64748b !important;
                margin-bottom: 0.8rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
                gap: 6px !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 12px !important;
                padding: 0.65rem 0.95rem !important;
                transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
                cursor: pointer !important;
                margin: 0 !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
                background: #f8fafc !important;
                border-color: #cbd5e1 !important;
                transform: translateX(4px) !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] div[aria-hidden="true"] {
                display: none !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
                background: linear-gradient(135deg, rgba(13, 148, 136, 0.08) 0%, rgba(2, 132, 199, 0.08) 100%) !important;
                border: 1px solid #0d9488 !important;
                box-shadow: 0 4px 12px rgba(13, 148, 136, 0.12) !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
                color: #0f766e !important;
                font-weight: 700 !important;
            }

            .dashboard-shell {
                max-width: 1350px;
                margin: 0 auto;
            }

            /* Light Studio Hero Card */
            .hero-card {
                position: relative;
                padding: 2.4rem 2.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
                border: 1px solid #e2e8f0;
                box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05);
                overflow: hidden;
                margin-bottom: 2rem;
                transition: all 0.3s ease;
            }

            .hero-card:hover {
                box-shadow: 0 16px 35px -5px rgba(13, 148, 136, 0.1);
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 0.4rem 0.95rem;
                border-radius: 999px;
                background: #ccfbf1;
                border: 1px solid #99f6e4;
                color: #0f766e;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-bottom: 1rem;
                text-transform: uppercase;
            }

            .section-title {
                font-size: 1.35rem;
                font-weight: 800;
                color: #0f172a;
                letter-spacing: -0.02em;
                margin: 1.6rem 0 0.3rem 0;
                display: flex;
                align-items: center;
                gap: 0.6rem;
            }

            .section-subtitle {
                color: #64748b;
                font-size: 0.92rem;
                margin-bottom: 1rem;
            }

            .dashboard-card {
                padding: 1.4rem 1.5rem;
                border-radius: 18px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            }

            .dashboard-card:hover {
                box-shadow: 0 16px 32px -6px rgba(13, 148, 136, 0.12);
                transform: translateY(-2px);
                border-color: #cbd5e1;
            }

            .mini-card {
                padding: 1.2rem;
                border-radius: 16px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
                transition: all 0.3s ease;
            }

            .mini-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.08);
                border-color: #0d9488;
            }

            .pipeline-step {
                padding: 1.2rem 1rem;
                border-radius: 16px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
                text-align: center;
                min-height: 125px;
                transition: all 0.3s ease;
            }

            .pipeline-step:hover {
                transform: translateY(-3px);
                box-shadow: 0 14px 28px -6px rgba(13, 148, 136, 0.15);
                border-color: #0d9488;
            }

            .pipeline-arrow {
                font-size: 1.8rem;
                color: #0d9488;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
                font-weight: 800;
            }

            .footer-bar {
                margin-top: 3rem;
                padding: 1rem 1.4rem;
                border-radius: 16px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                color: #64748b;
                font-size: 0.88rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
            }

            .sidebar-title {
                margin-top: 0.75rem;
                font-size: 1.35rem;
                font-weight: 800;
                line-height: 1.15;
                color: #0f172a;
            }

            .sidebar-chip {
                display: inline-block;
                padding: 0.28rem 0.65rem;
                border-radius: 999px;
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
                font-size: 0.78rem;
                font-weight: 600;
                color: #475569 !important;
            }

            /* Crisp Metric Cards Override */
            [data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-top: 3px solid #0d9488 !important;
                border-radius: 16px !important;
                padding: 1rem 1.2rem !important;
                box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
                transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }

            [data-testid="stMetric"]:hover {
                transform: translateY(-3px) !important;
                box-shadow: 0 14px 28px -6px rgba(13, 148, 136, 0.15) !important;
            }

            [data-testid="stMetricLabel"] p {
                font-size: 0.76rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.06em !important;
                color: #64748b !important;
            }

            [data-testid="stMetricValue"] div {
                font-size: 2rem !important;
                font-weight: 800 !important;
                color: #0f172a !important;
                letter-spacing: -0.03em !important;
            }

            div[data-testid="stFileUploader"] {
                background: #ffffff;
                border-radius: 18px;
                border: 2px dashed #0d9488;
                padding: 0.6rem;
                transition: all 0.3s ease;
            }

            div[data-testid="stFileUploader"]:hover {
                background: #f0fdf4;
            }

            [data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
            }

            .stButton > button {
                background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.65rem 1.4rem !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 14px rgba(13, 148, 136, 0.3) !important;
                transition: all 0.25s ease !important;
            }

            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 22px rgba(13, 148, 136, 0.4) !important;
            }

            .stDownloadButton > button {
                background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.65rem 1.4rem !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
                transition: all 0.25s ease !important;
            }

            .stDownloadButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 22px rgba(2, 132, 199, 0.4) !important;
            }

            hr {
                border-color: #e2e8f0 !important;
                margin: 1.2rem 0 !important;
            }

            /* Info tooltip expanders */
            .info-tooltip-expander .streamlit-expanderHeader {
                font-size: 0.82rem !important;
                font-weight: 600 !important;
                color: #475569 !important;
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 10px !important;
                padding: 0.4rem 0.8rem !important;
            }

            .info-tooltip-expander .streamlit-expanderContent {
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                border-top: none !important;
                border-radius: 0 0 10px 10px !important;
                padding: 0.6rem 0.8rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_image(image, caption: str | None = None) -> None:
    """Render image safely without deprecation warnings."""
    try:
        st.image(image, caption=caption, width="stretch")
    except (TypeError, ValueError):
        st.image(image, caption=caption, use_container_width=True)


def render_sidebar_brand() -> None:
    """Render the professional branded sidebar."""
    safe_image(logo_image())
    st.markdown(
        f"""
        <div style="margin-top:0.8rem;">
            <div class="sidebar-title">{APP_TITLE}</div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.35rem; line-height:1.4; font-weight:500;">
                {PROJECT_TITLE}
            </div>
            <div style="margin-top:0.8rem; display:flex; gap:0.35rem; flex-wrap:wrap;">
                <span class="sidebar-chip">Version {APP_VERSION}</span>
                <span class="sidebar-chip">{LICENSE_NAME}</span>
            </div>
            <div class="sidebar-profile-card">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <div class="sidebar-profile-avatar">HK</div>
                    <div>
                        <div class="sidebar-profile-name">{DEVELOPER_NAME}</div>
                        <div class="sidebar-profile-role">{DEVELOPER_ROLE}</div>
                    </div>
                </div>
                <div class="sidebar-status-badge">
                    <span class="status-dot"></span> Medical AI Systems Lead
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render a small footer with project metadata."""
    st.markdown(
        f"""
        <div class="footer-bar">
            <span><strong>GitHub:</strong> {GITHUB_PLACEHOLDER}</span>
            <span><strong>License:</strong> {LICENSE_NAME}</span>
            <span><strong>System Version:</strong> {APP_VERSION}</span>
            <span><strong>Developer:</strong> {DEVELOPER_NAME}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent section heading."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def info_tooltip(key: str, title: str, explanation: str, extended_explanation: str | None = None) -> None:
    """Render a compact ℹ️ info expander with an explanation.

    Args:
        key: Unique key for the expander (must be unique per page render).
        title: Short title describing what this tooltip explains.
        explanation: HTML-safe explanation text answering "What is this?"
            and "Why is it used in this project?".
        extended_explanation: Deep, easy-to-understand explanation of the core concept.
    """
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.markdown(
            f'<div style="color:#475569;line-height:1.7;font-size:0.86rem;margin-bottom:0.5rem;">'
            f"{explanation}</div>",
            unsafe_allow_html=True,
        )
        if extended_explanation:
            st.markdown(
                f"""
                <details style="margin-top:0.6rem;background:#f8fafc;padding:0.75rem;border-radius:8px;border:1px solid #e2e8f0;cursor:pointer;">
                    <summary style="font-weight:700;color:#0d9488;font-size:0.85rem;list-style:none;">
                        <span style="display:inline-flex;align-items:center;gap:4px;">
                            💡 Learn More <span style="font-size:0.75rem;">(Detailed Explanation)</span>
                        </span>
                    </summary>
                    <div style="margin-top:0.75rem;color:#334155;line-height:1.7;font-size:0.86rem;cursor:text;">
                        {extended_explanation}
                    </div>
                </details>
                """,
                unsafe_allow_html=True,
            )


def render_metric_cards(metrics: list[tuple[str, Any]]) -> None:
    """Render a responsive row of metric cards."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def count_supported_images(directory: Path) -> int:
    """Count supported image files in a folder recursively."""
    if not directory.is_dir():
        return 0

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_csv_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_history_dataframe() -> pd.DataFrame:
    """Load training history with backward-compatible fallback support."""
    for candidate in [LARGE_HISTORY_PATH, HISTORY_PATH, EXPERIMENT_RESULTS_PATH]:
        history = load_csv_dataframe(candidate)
        if not history.empty:
            return history
    return pd.DataFrame()


def load_active_learning_dataframe() -> pd.DataFrame:
    """Load uncertainty scores and normalize numeric columns."""
    frame = load_csv_dataframe(ACTIVE_LEARNING_PATH)
    if frame.empty:
        return frame

    for column in ["least_confidence", "entropy", "margin", "final_score"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return frame


def load_experiment_dataframe() -> pd.DataFrame:
    """Load the exported experiment results."""
    for candidate in [FINAL_METRICS_PATH, EXPERIMENT_RESULTS_PATH, LARGE_HISTORY_PATH]:
        frame = load_csv_dataframe(candidate)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def list_checkpoint_files() -> list[Path]:
    """Return all known checkpoint files if the checkpoint directory exists."""
    if not CHECKPOINT_DIR.is_dir():
        return []

    checkpoint_files = []
    for path in CHECKPOINT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() == ".pth":
            checkpoint_files.append(path)
    return sorted(checkpoint_files)


def list_prediction_directories() -> list[Path]:
    """Return all known prediction output folders."""
    directories = []
    for path in PREDICTION_DIRS:
        if path.is_dir():
            directories.append(path)
    return directories


def load_semi_supervised_report() -> dict[str, Any]:
    """Parse the semi-supervised text report into a dictionary."""
    if not SEMI_SUPERVISED_REPORT_PATH.exists():
        return {}

    result: dict[str, Any] = {}
    for line in SEMI_SUPERVISED_REPORT_PATH.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        try:
            if "." in value:
                result[key] = float(value)
            else:
                result[key] = int(value)
        except ValueError:
            result[key] = value
    return result


def best_metric_values(history: pd.DataFrame) -> tuple[float, float]:
    """Return the best Dice and IoU values from a training history."""
    if history.empty:
        return 0.0, 0.0

    dice = _safe_column_max(history, "Validation Dice")
    iou = _safe_column_max(history, "Validation IoU")
    return dice, iou


def best_epoch_value(history: pd.DataFrame) -> int:
    """Return the epoch with the best validation Dice."""
    if history.empty or "Epoch" not in history.columns:
        return 0

    if "Validation Dice" not in history.columns:
        epoch_values = pd.to_numeric(history["Epoch"], errors="coerce").dropna()
        if epoch_values.empty:
            return 0
        return int(epoch_values.iloc[-1])

    epochs = pd.to_numeric(history["Epoch"], errors="coerce")
    dice = pd.to_numeric(history["Validation Dice"], errors="coerce")
    valid_rows = pd.DataFrame({"Epoch": epochs, "Validation Dice": dice}).dropna()
    if valid_rows.empty:
        return 0

    best_index = valid_rows["Validation Dice"].idxmax()
    return int(valid_rows.loc[best_index, "Epoch"])


def final_epoch_value(history: pd.DataFrame) -> int:
    """Return the final epoch number from a history dataframe."""
    if history.empty or "Epoch" not in history.columns:
        return 0

    epoch_values = pd.to_numeric(history["Epoch"], errors="coerce").dropna()
    if epoch_values.empty:
        return 0
    return int(epoch_values.iloc[-1])


def average_training_loss(history: pd.DataFrame) -> float:
    """Return the average training loss from the available history."""
    return _safe_column_mean(history, "Training Loss")


def _safe_column_max(frame: pd.DataFrame, column: str) -> float:
    """Return a numeric max for a column or zero when unavailable."""
    if column not in frame.columns:
        return 0.0

    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.max())


def _safe_column_mean(frame: pd.DataFrame, column: str) -> float:
    """Return a numeric mean for a column or zero when unavailable."""
    if column not in frame.columns:
        return 0.0

    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def render_home_page() -> None:
    """Render the landing page for the dashboard."""
    import torch as _torch
    from configs.config import CHECKPOINT_PATH as _CKPT

    dataset_config = get_dataset_configuration()
    dataset_root = dataset_config.dataset_root
    labeled_images = count_supported_images(dataset_root / "images")
    unlabeled_images = count_supported_images(dataset_root / "unlabeled")
    pseudo_labels = count_supported_images(dataset_root / "pseudo_masks")
    train_images = count_supported_images(dataset_root / "train" / "images")
    val_images = count_supported_images(dataset_root / "val" / "images")

    history = load_history_dataframe()
    best_dice, best_iou = best_metric_values(history)

    _dev = "NVIDIA GPU (CUDA)" if _torch.cuda.is_available() else "CPU"
    _gpu = ""
    if _torch.cuda.is_available():
        try:
            _gpu = _torch.cuda.get_device_name(0)
        except Exception:
            _gpu = "CUDA GPU"
    _ckpt_file = _CKPT / "best_model.pth"
    _ckpt_ok = _ckpt_file.is_file()
    _ckpt_mb = _ckpt_file.stat().st_size / 1e6 if _ckpt_ok else 0

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">Medical AI Research Dashboard</div>
            <h1 style="margin:0 0 0.5rem 0; font-size:2.1rem; font-weight:800; color:#0f172a;">
                Medical AI Segmentation
            </h1>
            <p style="font-size:1.04rem; color:#475569; max-width:900px; line-height:1.7; margin-bottom:0.6rem;">
                {PROJECT_TITLE}
            </p>
            <p style="font-size:0.92rem; color:#64748b; max-width:900px; line-height:1.7; margin-bottom:1rem;">
                Research objective: reduce manual annotation effort while maintaining
                segmentation quality through active learning and pseudo-labelling.
            </p>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                <span style="display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.75rem;border-radius:999px;background:#f0fdf4;border:1px solid #bbf7d0;color:#15803d;font-size:0.75rem;font-weight:600;">● {_dev}</span>
                <span style="display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.75rem;border-radius:999px;background:{'#f0fdf4' if _ckpt_ok else '#fef2f2'};border:1px solid {'#bbf7d0' if _ckpt_ok else '#fecaca'};color:{'#15803d' if _ckpt_ok else '#dc2626'};font-size:0.75rem;font-weight:600;">{'✅' if _ckpt_ok else '❌'} Checkpoint {'Ready' if _ckpt_ok else 'Missing'}</span>
                <span style="display:inline-flex;align-items:center;gap:5px;padding:0.3rem 0.75rem;border-radius:999px;background:#f0f9ff;border:1px solid #bae6fd;color:#0369a1;font-size:0.75rem;font-weight:600;">U-Net Architecture</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Training Results ---
    render_section_header("Training Results", "Best metrics from the most recent training run.")
    _mr = st.columns(5)
    with _mr[0]:
        st.metric("Best Dice", "N/A" if history.empty else f"{best_dice:.4f}")
    with _mr[1]:
        st.metric("Best IoU", "N/A" if history.empty else f"{best_iou:.4f}")
    with _mr[2]:
        st.metric("Best Epoch", best_epoch_value(history))
    with _mr[3]:
        st.metric("Total Epochs", final_epoch_value(history))
    with _mr[4]:
        st.metric("Avg Loss", "N/A" if history.empty else f"{average_training_loss(history):.4f}")

    info_tooltip(
        "home_train_metrics",
        "Training Metrics",
        "<strong>Best Dice</strong> — Highest Dice coefficient achieved on validation data. "
        "Measures overlap between predicted and ground truth masks (1.0 = perfect).<br><br>"
        "<strong>Best IoU</strong> — Highest Intersection over Union achieved on validation data. "
        "IoU = |Pred ∩ GT| / |Pred ∪ GT|.<br><br>"
        "<strong>Best Epoch</strong> — The training epoch that produced the highest validation Dice.<br><br>"
        "<strong>Total Epochs</strong> — Total number of training iterations completed.<br><br>"
        "<strong>Avg Loss</strong> — Average training loss (BCE + Dice) across all epochs. "
        "Lower values indicate better optimization.",
    )

    # --- System Status ---
    render_section_header("System Status", "GPU, checkpoint, and model information.")
    _sr = st.columns(4)
    for _col, (_lbl, _val) in zip(_sr, [
        ("GPU Device", _gpu if _gpu else _dev),
        ("Checkpoint Size", f"{_ckpt_mb:.1f} MB"),
        ("Model Params", "31.0 M"),
        ("Input Resolution", "256 × 256"),
    ]):
        with _col:
            st.markdown(
                f'<div class="mini-card"><div style="color:#64748b;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">{_lbl}</div><div style="font-size:1.05rem;font-weight:800;margin-top:0.3rem;color:#0f172a;">{_val}</div></div>',
                unsafe_allow_html=True,
            )

    info_tooltip(
        "home_system_info",
        "System Status",
        "<strong>GPU Device</strong> — The compute device used for training and inference. "
        "CUDA GPUs provide significant acceleration for deep learning workloads.<br><br>"
        "<strong>Checkpoint Size</strong> — On-disk size of the best saved model weights.<br><br>"
        "<strong>Model Params</strong> — Total number of learnable parameters in the U-Net model.<br><br>"
        "<strong>Input Resolution</strong> — All images are resized to this spatial dimension before processing.",
    )

    # --- Dataset Statistics ---
    render_section_header("Dataset Statistics", "Quick readout from the prepared project folders.")
    _ds_cols = st.columns(7)
    ds_items = [
        ("Labeled Images", labeled_images),
        ("Train Images", train_images),
        ("Val Images", val_images),
        ("Unlabeled Images", unlabeled_images),
        ("Pseudo Labels", pseudo_labels),
        ("Best Dice", "N/A" if history.empty else f"{best_dice:.4f}"),
        ("Best IoU", "N/A" if history.empty else f"{best_iou:.4f}"),
    ]
    for _col, (_lbl, _val) in zip(_ds_cols, ds_items):
        with _col:
            st.metric(_lbl, _val)

    info_tooltip(
        "home_dataset_info",
        "Dataset Statistics",
        "<strong>Labeled Images</strong> — Images with manually annotated ground truth masks.<br><br>"
        "<strong>Train/Val Images</strong> — Images in the training and validation splits respectively. "
        "The validation set is used to evaluate model performance on unseen data.<br><br>"
        "<strong>Unlabeled Images</strong> — Images without ground truth masks, used for "
        "active learning and pseudo-labelling.<br><br>"
        "<strong>Pseudo Labels</strong> — Machine-generated masks for unlabeled images using "
        "high-confidence model predictions.",
    )

    # --- Learning Curves ---
    if not history.empty:
        render_section_header("Learning Curves", "Training loss and validation metrics across epochs.")
        _tabs = st.tabs(["Training Loss", "Validation Dice", "Validation IoU"])
        with _tabs[0]:
            if {"Epoch", "Training Loss"}.issubset(history.columns):
                st.line_chart(history.set_index("Epoch")["Training Loss"])
            else:
                st.info("Training loss data is unavailable.")
        with _tabs[1]:
            if {"Epoch", "Validation Dice"}.issubset(history.columns):
                st.line_chart(history.set_index("Epoch")["Validation Dice"])
            else:
                st.info("Validation Dice data is unavailable.")
        with _tabs[2]:
            if {"Epoch", "Validation IoU"}.issubset(history.columns):
                st.line_chart(history.set_index("Epoch")["Validation IoU"])
            else:
                st.info("Validation IoU data is unavailable.")

        info_tooltip(
            "home_curves_info",
            "Learning Curves",
            "<strong>Training Loss</strong> — Measures how different the model's predictions "
            "are from the ground truth masks during optimization. Generally, lower loss indicates "
            "better optimization, but this does not always guarantee better real-world "
            "segmentation performance.<br><br>"
            "<strong>Validation Dice / IoU</strong> — These curves show segmentation "
            "performance on unseen validation data. Increasing values over epochs indicate "
            "the model is learning meaningful features.",
        )

    # --- Workflow Diagram ---
    render_section_header("Research Workflow", "End-to-end research pipeline overview.")
    _workflow_steps = [
        ("Dataset", "Labeled pairs, unlabeled scans, train/val splits",
         "The project uses paired medical images and segmentation masks. "
         "Data is split into training and validation sets for supervised learning."),
        ("Supervised Training", "U-Net optimization with BCE + Dice loss",
         "The U-Net model is trained on labelled image-mask pairs using a combined "
         "binary cross-entropy and Dice loss function."),
        ("Uncertainty Estimation", "Pixel-wise model confidence analysis",
         "The trained model is used to compute uncertainty scores on unlabeled images, "
         "measuring how confident the model is about each prediction."),
        ("Active Learning", "Rank and select most informative samples",
         "Unlabeled samples are ranked by uncertainty. The most uncertain samples are "
         "selected for manual annotation, reducing total annotation effort."),
        ("Pseudo-Labelling", "Generate masks for high-confidence predictions",
         "High-confidence model predictions on unlabeled data are used as pseudo ground "
         "truth masks to expand the training set without manual annotation."),
        ("Retraining", "Merge pseudo labels and retrain the model",
         "The model is retrained on the expanded dataset (original labels + pseudo labels) "
         "to improve segmentation performance."),
        ("Evaluation", "Dice, IoU, precision, recall, F1 metrics",
         "The final model is evaluated using standard medical image segmentation metrics "
         "on the held-out validation set."),
    ]

    for idx in range(0, len(_workflow_steps), 2):
        cols = st.columns([1, 0.15, 1] if idx + 1 < len(_workflow_steps) else [1])
        with cols[0]:
            _t, _s, _exp = _workflow_steps[idx]
            st.markdown(
                f'<div class="pipeline-step"><div style="font-weight:800;font-size:1.03rem;color:#0f172a;">{_t}</div><div style="margin-top:0.55rem;color:#475569;line-height:1.5;">{_s}</div></div>',
                unsafe_allow_html=True,
            )
            info_tooltip(f"wf_step_{idx}", _t, _exp)
        if idx + 1 < len(_workflow_steps):
            with cols[1]:
                st.markdown('<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True)
            with cols[2]:
                _t, _s, _exp = _workflow_steps[idx + 1]
                st.markdown(
                    f'<div class="pipeline-step"><div style="font-weight:800;font-size:1.03rem;color:#0f172a;">{_t}</div><div style="margin-top:0.55rem;color:#475569;line-height:1.5;">{_s}</div></div>',
                    unsafe_allow_html=True,
                )
                info_tooltip(
                    f"wf_step_{idx+1}", 
                    _t, 
                    _exp,
                    "In the context of artificial intelligence, a 'workflow' or 'pipeline' is simply "
                    "the step-by-step process we follow to build and improve a model. Think of it like "
                    "a recipe. First, we gather our ingredients (Data), then we bake our cake (Train "
                    "the model). After tasting it, we might realise it needs more sugar, so we figure "
                    "out exactly what's missing (Uncertainty Estimation), get those specific ingredients "
                    "(Active Learning), and bake it again to make it even better."
                )

    # --- Model Architecture ---
    render_section_header("Model Architecture", "The project uses a compact U-Net for binary segmentation.")
    _mc = st.columns(4)
    for _col, (_lbl, _val) in zip(_mc, [
        ("Architecture", "U-Net"),
        ("Input Channels", "3 (RGB)"),
        ("Output Channels", "1 (Binary)"),
        ("Loss Function", "BCE + Dice"),
    ]):
        with _col:
            st.markdown(
                f'<div class="mini-card"><div style="color:#64748b;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">{_lbl}</div><div style="font-size:1.1rem;font-weight:800;margin-top:0.25rem;color:#0f172a;">{_val}</div></div>',
                unsafe_allow_html=True,
            )

    info_tooltip(
        "home_arch_info",
        "Model Architecture",
        "<strong>U-Net</strong> — A convolutional neural network designed for biomedical image "
        "segmentation. It uses an encoder-decoder structure with skip connections to preserve "
        "spatial detail.<br><br>"
        "<strong>3 RGB Input Channels</strong> — The model accepts standard RGB images.<br><br>"
        "<strong>1 Binary Output</strong> — The model produces a single-channel output where each "
        "pixel value represents the probability of belonging to the target class.<br><br>"
        "<strong>BCE + Dice Loss</strong> — A combined loss function: Binary Cross-Entropy handles "
        "per-pixel classification while Dice Loss optimises for overall segmentation overlap.",
        "An 'Architecture' in AI is like the blueprint of a building. The U-Net is a specific blueprint "
        "famous in medical imaging because of its 'U' shape. It first shrinks the image down to understand "
        "the 'big picture' (like realising it's looking at a brain), and then expands it back up to "
        "pinpoint exactly where things are (like drawing a boundary around a tumor). "
        "The 'Channels' are just the colors it sees (Red, Green, Blue) and the single output it produces "
        "(Black for background, White for tumor). Finally, the 'Loss Function' is how the model is graded; "
        "it penalises the model when it makes mistakes, forcing it to learn and improve."
    )

    # --- Project Overview ---
    render_section_header("Project Overview", "Research capabilities demonstrated in this system.")
    overview_left, overview_right = st.columns([1.2, 0.8], gap="large")
    with overview_left:
        st.markdown(
            """
            <div class="dashboard-card">
                <strong>What this project demonstrates</strong>
                <ul style="margin-top:0.7rem; line-height:1.8; color:#475569;">
                    <li>Supervised segmentation with a U-Net backbone.</li>
                    <li>Training history and experiment tracking.</li>
                    <li>Uncertainty-based active learning on unlabeled scans.</li>
                    <li>Pseudo-label generation for semi-supervised expansion.</li>
                    <li>Evaluation and visualization of segmentation results.</li>
                    <li>Explainable AI with GradCAM visualizations.</li>
                    <li>CPU vs GPU benchmarking and comparison.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_right:
        st.markdown(
            """
            <div class="dashboard-card">
                <strong>Research workflow</strong>
                <p style="margin-top:0.6rem; color:#475569; line-height:1.7;">
                    The dashboard connects the full experimental loop from
                    dataset preparation to evaluation so each stage can be
                    explained clearly in a demo or viva.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_page() -> None:
    """Render the about section for the dashboard."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">About the Research System</div>
            <h1 style="margin:0 0 0.4rem 0;">Project Architecture & Contributions</h1>
            <p style="margin:0; color:#334155; line-height:1.7; max-width: 980px;">
                This dashboard is designed to present the full medical image
                segmentation pipeline in a polished, research-grade format.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("Project Architecture")
    info_tooltip(
        "about_arch_info",
        "Project Architecture",
        "The project is organised into modular directories, each responsible for a "
        "specific part of the research pipeline. This separation makes the codebase "
        "easier to understand, test, and extend.",
        "Just like a well-organised office has different departments for different tasks "
        "(HR, Finance, IT), a professional software project separates its code into different "
        "folders. 'Models' is where the AI's brain is kept. 'Dataset' is where it studies. "
        "'Evaluation' is where it takes its exams. Keeping things separated like this means "
        "if we want to change how the AI learns, we only have to look in one specific folder, "
        "rather than searching through thousands of lines of code."
    )
    st.markdown(
        """
        <div class="dashboard-card">
            <ul style="margin:0; line-height:1.9; color:#334155;">
                <li><strong>configs/</strong> - project paths, image size, and training settings.</li>
                <li><strong>dataset/</strong> - labeled, unlabeled, pseudo-label, train, and validation data.</li>
                <li><strong>models/</strong> - U-Net architecture and reusable blocks.</li>
                <li><strong>utils/</strong> - dataset loading, metrics, losses, plotting, splitting, and helpers.</li>
                <li><strong>active_learning/</strong> - uncertainty scoring and query selection.</li>
                <li><strong>semi_supervised/</strong> - pseudo-label generation and retraining preparation.</li>
                <li><strong>evaluation/</strong> - experiment summaries and publication-style plots.</li>
                <li><strong>dashboard/</strong> - Streamlit user interface for demos and exploration.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("Folder Structure")
    st.markdown(
        """
        <div class="dashboard-card">
            <pre style="margin:0; white-space:pre-wrap; color:#334155; line-height:1.6;">
Medical_Segmentation/
├── app.py
├── configs/
├── dataset/
├── models/
├── utils/
├── active_learning/
├── semi_supervised/
├── evaluation/
├── outputs/
└── dashboard/
            </pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        render_section_header("Technologies Used")
        info_tooltip(
            "about_tech_info",
            "Technologies",
            "<strong>Python</strong> — Primary programming language.<br>"
            "<strong>PyTorch</strong> — Deep learning framework for model training and inference.<br>"
            "<strong>OpenCV</strong> — Image loading, resizing, and preprocessing.<br>"
            "<strong>Pandas</strong> — Data manipulation and experiment tracking.<br>"
            "<strong>Matplotlib</strong> — Plotting and visualization.<br>"
            "<strong>Streamlit</strong> — Web-based dashboard framework.",
            "These are the tools used to build this entire system. Python is the core language holding "
            "everything together. PyTorch is the heavy-lifting engine that handles the complex math "
            "required for the AI to learn. OpenCV is like a digital magnifying glass used to resize and "
            "adjust the medical images so the AI can read them properly. Pandas acts like a highly advanced "
            "Excel spreadsheet to keep track of numbers and scores. Matplotlib draws the charts, and "
            "Streamlit is what creates the beautiful website you are looking at right now!"
        )
        st.markdown(
            """
            <div class="dashboard-card">
                <ul style="margin:0; line-height:1.9; color:#334155;">
                    <li>Python</li>
                    <li>PyTorch</li>
                    <li>OpenCV</li>
                    <li>Pandas</li>
                    <li>Matplotlib</li>
                    <li>Streamlit</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        render_section_header("Research Contributions")
        info_tooltip(
            "about_contrib_info",
            "Research Contributions",
            "This project demonstrates a complete semi-supervised active learning "
            "pipeline for medical image segmentation, combining supervised training, "
            "uncertainty-based sample selection, and pseudo-label generation to reduce "
            "annotation costs while maintaining segmentation quality.",
            "In medical AI, the biggest problem is that doctors are very busy. Having a doctor "
            "manually draw boundaries around tumors on thousands of images is expensive and slow. "
            "This project solves that problem. It trains an AI on a small set of doctor-labeled images, "
            "then has the AI look at a massive pile of unlabeled images. The AI picks out only the "
            "most confusing images for the doctor to look at (Active Learning), and automatically "
            "labels the easy ones itself (Pseudo-labeling). This creates a highly accurate system "
            "with a fraction of the manual effort."
        )
        st.markdown(
            """
            <div class="dashboard-card">
                <ul style="margin:0; line-height:1.9; color:#334155;">
                    <li>End-to-end binary medical image segmentation workflow.</li>
                    <li>Uncertainty-based active learning for annotation efficiency.</li>
                    <li>Pseudo-labeling for semi-supervised learning.</li>
                    <li>Experiment logging, visualization, and evaluation support.</li>
                    <li>A polished dashboard for demonstrations and review.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_research_pipeline_page() -> None:
    """Render the end-to-end research pipeline view."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Research Pipeline</div>
            <h1 style="margin:0 0 0.4rem 0;">From Dataset to Evaluation</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                A visual walkthrough of how the project modules interact across
                the full research workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        ("1. Dataset", "Load labeled and unlabeled medical images."),
        ("2. Training", "Train the U-Net on paired images and masks."),
        ("3. Active Learning", "Rank unlabeled scans by uncertainty."),
        ("4. Pseudo Labeling", "Generate masks for high-confidence samples."),
        ("5. Retraining", "Merge pseudo labels back into the dataset."),
        ("6. Evaluation", "Measure Dice, IoU, precision, recall, and F1."),
    ]

    for index in range(0, len(steps), 2):
        left_col, arrow_col, right_col = st.columns([1, 0.2, 1], gap="medium")
        with left_col:
            title, subtitle = steps[index]
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div style="font-weight:800; font-size:1.02rem;">{title}</div>
                    <div style="margin-top:0.5rem; color:#475569; line-height:1.6;">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with arrow_col:
            st.markdown(
                '<div class="pipeline-arrow">&rarr;</div>',
                unsafe_allow_html=True,
            )
        with right_col:
            if index + 1 < len(steps):
                title, subtitle = steps[index + 1]
                st.markdown(
                    f"""
                    <div class="pipeline-step">
                        <div style="font-weight:800; font-size:1.02rem;">{title}</div>
                        <div style="margin-top:0.5rem; color:#475569; line-height:1.6;">{subtitle}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="pipeline-step"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="dashboard-card" style="margin-top:1rem;">
            <strong>Why this design works for research demos</strong>
            <p style="margin-top:0.6rem; color:#334155; line-height:1.75;">
                Each stage is separated visually, but they remain connected
                through the existing codebase. That makes the system easier to
                explain during internships, presentations, and reviews.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_tooltip(
        "rp_design_info",
        "Pipeline Design",
        "The research pipeline follows the standard semi-supervised active learning "
        "loop: train on labelled data, estimate uncertainty on unlabelled data, "
        "select the most informative samples for annotation or pseudo-labelling, "
        "then retrain with the expanded dataset. Each stage feeds into the next.",
        "Think of this pipeline as a student in a classroom. First, the student learns "
        "from a textbook (Training on Dataset). Then, the student takes a practice test "
        "(Evaluation). For the questions the student found really difficult, they ask the "
        "teacher for help (Active Learning). For the questions they found super easy, they "
        "write their own flashcards to study later (Pseudo Labeling). Finally, the student "
        "studies everything again (Retraining) to become even smarter."
    )


def _numeric_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first usable numeric series from a list of column names."""
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").dropna()
    return pd.Series(dtype=float)


def find_confusion_matrix_artifact() -> Path | None:
    """Locate a confusion matrix artifact if one exists."""
    candidates = [
        OUTPUT_PATH / "confusion_matrix.png",
        OUTPUT_PATH / "confusion_matrix.jpg",
        OUTPUT_PATH / "confusion_matrix.jpeg",
        OUTPUT_PATH / "confusion_matrix.csv",
        PLOTS_DIR / "confusion_matrix.png",
        PLOTS_DIR / "confusion_matrix.jpg",
        PLOTS_DIR / "confusion_matrix.jpeg",
        PLOTS_DIR / "confusion_matrix.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def render_training_results_page() -> None:
    """Render the training history and learning curves page."""
    history = load_history_dataframe()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Training Results</div>
            <h1 style="margin:0 0 0.4rem 0;">Learning Curves & Checkpoints</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Monitor training loss, validation Dice, and validation IoU from
                the saved experiment history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history.empty:
        st.info("No training history found in outputs/history.csv.")
        return

    for column in ["Epoch", "Training Loss", "Validation Dice", "Validation IoU"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    best_epoch = best_epoch_value(history)
    best_dice, best_iou = best_metric_values(history)
    avg_loss = average_training_loss(history)
    final_epoch = final_epoch_value(history)

    render_section_header("Training Summary")
    render_metric_cards(
        [
            ("Best Epoch", best_epoch),
            ("Best Dice", f"{best_dice:.4f}" if best_dice else "N/A"),
            ("Best IoU", f"{best_iou:.4f}" if best_iou else "N/A"),
            ("Average Loss", f"{avg_loss:.4f}" if avg_loss else "N/A"),
            ("Final Epoch", final_epoch),
        ]
    )

    info_tooltip(
        "tr_summary_info",
        "Training Summary Metrics",
        "<strong>Best Dice</strong> — Dice = 2|Pred ∩ GT| / (|Pred| + |GT|). "
        "Range: 0 (no overlap) to 1 (perfect overlap). Higher is better.<br><br>"
        "<strong>Best IoU</strong> — IoU = |Pred ∩ GT| / |Pred ∪ GT|. "
        "Range: 0 to 1. Higher is better.<br><br>"
        "<strong>Average Loss</strong> — Training loss measures how different the model's "
        "predictions are from the ground truth masks during optimization. Lower loss "
        "generally indicates better optimization, but does not always guarantee better "
        "real-world segmentation.<br><br>"
        "<strong>Best/Final Epoch</strong> — The epoch with the best validation Dice, "
        "and the last epoch completed.",
        "When an AI is learning, we need mathematical ways to give it a score out of 100%. "
        "<strong>Dice</strong> and <strong>IoU (Intersection over Union)</strong> are simply two different ways of calculating "
        "how perfectly the AI's predicted drawing matches the doctor's real drawing. A score of 1.0 "
        "means a perfect match, and 0.0 means they missed completely. "
        "<strong>Loss</strong> is the opposite — it's a measure of how many mistakes the AI made. We want "
        "Loss to go down, and Dice/IoU to go up. An <strong>Epoch</strong> is just one complete read-through "
        "of the training data, like reading a textbook from cover to cover."
    )

    render_section_header("Experiment History")
    st.dataframe(history, use_container_width=True, hide_index=True)
    info_tooltip(
        "tr_history_info",
        "Experiment History",
        "The full epoch-by-epoch record of training loss and validation metrics. "
        "Each row represents one training epoch.",
        "This table is the AI's report card over time. As you look down the rows, you should "
        "generally see the AI getting smarter (metrics going up) and making fewer mistakes "
        "(loss going down) with each passing epoch."
    )

    render_section_header("Learning Curves")
    info_tooltip(
        "tr_curves_info",
        "Learning Curves",
        "<strong>Training Loss</strong> — Should decrease over epochs, indicating the model "
        "is learning to match predictions to ground truth.<br><br>"
        "<strong>Validation Dice</strong> — Measures segmentation quality on unseen data. "
        "Increasing curves indicate generalisation.<br><br>"
        "<strong>Validation IoU</strong> — Similar to Dice but penalises false positives "
        "more heavily. Both Dice and IoU are evaluation metrics, NOT confidence scores.",
        "Learning curves are the best way to visualize how well the AI is learning. "
        "Ideally, the <strong>Loss curve</strong> should look like a slide going down, meaning mistakes are dropping. "
        "The <strong>Dice and IoU curves</strong> should look like a hill going up, meaning accuracy is rising. "
        "If the curves suddenly jump around wildly or start going in the wrong direction, it tells "
        "engineers that the AI is struggling to learn or is simply memorizing the data instead of understanding it."
    )
    tabs = st.tabs(["Training Loss", "Validation Dice", "Validation IoU"])
    with tabs[0]:
        if {"Epoch", "Training Loss"}.issubset(history.columns):
            st.line_chart(history.set_index("Epoch")["Training Loss"])
        else:
            st.info("Training loss data is unavailable.")
    with tabs[1]:
        if {"Epoch", "Validation Dice"}.issubset(history.columns):
            st.line_chart(history.set_index("Epoch")["Validation Dice"])
        else:
            st.info("Validation Dice data is unavailable.")
    with tabs[2]:
        if {"Epoch", "Validation IoU"}.issubset(history.columns):
            st.line_chart(history.set_index("Epoch")["Validation IoU"])
        else:
            st.info("Validation IoU data is unavailable.")


def render_active_learning_page() -> None:
    """Render the active learning dashboard page."""
    frame = load_active_learning_dataframe()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">🎯 Active Learning</div>
            <h1 style="margin:0 0 0.4rem 0; font-weight:800; color:#0f172a;">Uncertainty Ranking Dashboard</h1>
            <p style="margin:0; color:#475569; line-height:1.7;">
                Review uncertainty scores and inspect the most informative
                unlabeled samples for annotation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_tooltip(
        "al_overview_info",
        "Active Learning",
        "Active learning attempts to identify the unlabeled samples from which "
        "the model can learn most effectively, reducing the amount of manual "
        "annotation required. Samples are ranked by uncertainty — the most "
        "uncertain samples are the ones the model would benefit most from having labelled.",
        "Imagine you are studying for a test. You wouldn't waste time asking your teacher to explain "
        "flashcards you already know perfectly. You would only ask for help on the flashcards that "
        "confuse you the most. Active Learning does exactly this for AI. Instead of forcing a human "
        "doctor to label thousands of random medical images, the AI looks at the images, decides which "
        "ones it finds the most confusing (high uncertainty), and asks the doctor to label <i>only those</i>. "
        "This saves massive amounts of time and money."
    )

    if frame.empty:
        st.info("No active learning results were found in outputs/active_learning_scores.csv.")
        return

    if "filename" not in frame.columns:
        st.warning("The active learning CSV does not contain a filename column.")
        return

    score_columns = [
        column
        for column in ["final_score", "entropy", "least_confidence", "margin"]
        if column in frame.columns
    ]
    if not score_columns:
        st.warning("No uncertainty score columns were found in the CSV file.")
        return

    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        score_column = st.selectbox("Sort by", score_columns, index=0)
        info_tooltip(
            "al_sort_info",
            "Sort By",
            "Select the uncertainty metric to rank samples by. Available metrics depend "
            "on the active learning implementation:<br><br>"
            "<strong>final_score</strong> — Combined uncertainty score.<br>"
            "<strong>entropy</strong> — Information-theoretic uncertainty measure.<br>"
            "<strong>least_confidence</strong> — 1 minus the maximum predicted probability.<br>"
            "<strong>margin</strong> — Difference between the two highest class probabilities.",
            "There are different mathematical ways an AI can say 'I am confused'. "
            "<strong>Least Confidence</strong> means the AI's best guess was still a very weak guess. "
            "<strong>Margin</strong> means the AI is completely torn between two different choices (like a 51% vs 49% split). "
            "<strong>Entropy</strong> is a measure of total chaos—meaning the AI has absolutely no idea what it is looking at."
        )
        max_items = max(1, min(20, len(frame)))
        top_k = st.slider("Top samples to display", 1, max_items, min(5, max_items))

        frame = frame.copy()
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce").fillna(0)
        frame = frame.sort_values(by=score_column, ascending=False)

        render_section_header("📊 Selection Summary")
        render_metric_cards([
            ("Total Samples", len(frame)),
            ("Top-K", top_k),
        ])

        # Stats cards
        if len(frame) > 0:
            _scores = frame[score_column]
            _scols = st.columns(2)
            with _scols[0]:
                st.metric("Mean Score", f"{_scores.mean():.4f}")
            with _scols[1]:
                st.metric("Std Dev", f"{_scores.std():.4f}")

    with right:
        render_section_header("🔝 Top Uncertain Samples")
        top_samples = frame.head(top_k)[["filename", score_column]]
        for _, row in top_samples.iterrows():
            _pct = float(row[score_column]) * 100
            st.markdown(
                f"""
                <div class="mini-card" style="margin-bottom:0.7rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-weight:800; color:#0f172a;">{row['filename']}</div>
                        <div style="font-weight:700; color:#0f766e;">{float(row[score_column]):.4f}</div>
                    </div>
                    <div style="margin-top:0.4rem;background:#e2e8f0;border-radius:999px;height:6px;overflow:hidden;">
                        <div style="width:{min(_pct, 100):.1f}%;height:100%;background:linear-gradient(90deg,#0d9488,#0284c7);border-radius:999px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Uncertainty Distribution Histogram
    render_section_header("📈 Uncertainty Distribution")
    _hist_tabs = st.tabs(["Histogram", "Score Distribution"])
    with _hist_tabs[0]:
        st.bar_chart(frame.head(top_k).set_index("filename")[score_column])
    with _hist_tabs[1]:
        if len(frame) > 1:
            _hist_data = pd.DataFrame({
                score_column.replace("_", " ").title(): frame[score_column].values,
            })
            st.bar_chart(_hist_data)
        else:
            st.info("Not enough data points for a distribution chart.")

    # Entropy / Confidence histogram if columns exist
    if "entropy" in frame.columns and "least_confidence" in frame.columns:
        render_section_header("🔬 Entropy vs Confidence")
        _ec = st.columns(2)
        with _ec[0]:
            st.markdown('<div class="dashboard-card"><strong>Entropy Distribution</strong></div>', unsafe_allow_html=True)
            st.bar_chart(frame.sort_values("entropy", ascending=False).head(top_k).set_index("filename")["entropy"])
        with _ec[1]:
            st.markdown('<div class="dashboard-card"><strong>Confidence Distribution</strong></div>', unsafe_allow_html=True)
            st.bar_chart(frame.sort_values("least_confidence", ascending=False).head(top_k).set_index("filename")["least_confidence"])

    render_section_header("Uncertainty Table")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    info_tooltip(
        "al_table_info",
        "Uncertainty Table",
        "The complete table of all unlabeled samples with their computed uncertainty "
        "scores. Higher scores indicate samples where the model is least confident "
        "and would benefit most from manual annotation.",
        "This is the AI's actual 'list of questions' for the human doctor. The images at the very "
        "top of this table are the ones causing the AI the most confusion. By sending these specific "
        "images to a medical expert for labeling, the AI will learn the fastest."
    )

    # CSV Download
    render_section_header("Export")
    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Uncertainty Scores CSV",
        data=csv_data,
        file_name="active_learning_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )
    info_tooltip(
        "al_export_info",
        "Export",
        "Download the uncertainty scores as a CSV file for offline analysis, "
        "reporting, or integration with external annotation tools.",
        "This button allows you to download the AI's confusion list as a spreadsheet. "
        "In a real hospital or research lab, this spreadsheet would be sent directly to "
        "the software that doctors use to draw the tumor labels."
    )


def render_evaluation_page() -> None:
    """Render the experiment evaluation page."""
    frame = load_experiment_dataframe()
    history = load_history_dataframe()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">📈 Evaluation</div>
            <h1 style="margin:0 0 0.4rem 0; font-weight:800; color:#0f172a;">Experiment Summary &amp; Metrics</h1>
            <p style="margin:0; color:#475569; line-height:1.7;">
                Review the final metrics, learning curves, and available
                artifacts from the completed experiments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if frame.empty:
        st.info("No experiment metrics were found in the outputs folder.")
    else:
        render_section_header("Experiment Summary")
        best_dice_series = _numeric_series(frame, ["Validation Dice", "validation_dice", "dice"])
        best_iou_series = _numeric_series(frame, ["Validation IoU", "validation_iou", "iou"])
        precision_series = _numeric_series(frame, ["Precision", "precision"])
        recall_series = _numeric_series(frame, ["Recall", "recall"])
        f1_series = _numeric_series(frame, ["F1 Score", "f1_score"])
        epoch_series = _numeric_series(frame, ["Epoch", "epoch"])

        summary_values = [
            ("Best Dice", best_dice_series.max() if not best_dice_series.empty else None),
            ("Best IoU", best_iou_series.max() if not best_iou_series.empty else None),
            ("Precision", precision_series.max() if not precision_series.empty else None),
            ("Recall", recall_series.max() if not recall_series.empty else None),
            ("F1 Score", f1_series.max() if not f1_series.empty else None),
            ("Final Epoch", epoch_series.iloc[-1] if not epoch_series.empty else None),
        ]
        metrics = st.columns(6)
        for column, (label, value) in zip(metrics, summary_values):
            with column:
                if value is None or pd.isna(value):
                    display_value = "N/A"
                elif "Epoch" in label:
                    display_value = int(value)
                else:
                    display_value = f"{float(value):.4f}"
                st.metric(label, display_value)

        info_tooltip(
            "eval_summary_info",
            "Evaluation Metrics",
            "<strong>Dice</strong> — Dice = 2|Pred ∩ GT| / (|Pred| + |GT|). "
            "1.0 = perfect overlap, 0.0 = no overlap.<br><br>"
            "<strong>IoU</strong> — IoU = |Pred ∩ GT| / |Pred ∪ GT|. "
            "1.0 = perfect, 0.0 = no overlap.<br><br>"
            "<strong>Precision</strong> — Of all pixels predicted as foreground, "
            "what fraction actually belongs to the target class.<br><br>"
            "<strong>Recall</strong> — Of all actual foreground pixels, what fraction "
            "was correctly detected.<br><br>"
            "<strong>F1 Score</strong> — Harmonic mean of precision and recall.<br><br>"
            "These are <em>evaluation metrics</em> computed against ground truth, "
            "not measures of model confidence.",
            "Once training is completely finished, we have to give the AI a final exam. "
            "<strong>Precision</strong> answers the question: 'When the AI claimed there was a tumor, how often was it right?' "
            "<strong>Recall</strong> answers the question: 'Out of all the real tumors that existed, how many did the AI successfully find?' "
            "Sometimes an AI can be too aggressive (finding tumors everywhere, good recall but terrible precision) or too cautious "
            "(only guessing when it's 100% sure, missing a lot of tumors). "
            "The <strong>F1 Score, Dice, and IoU</strong> are balanced scores that ensure the AI is being both accurate and thorough."
        )

        render_section_header("📋 Experiment Results")
        st.dataframe(frame, use_container_width=True, hide_index=True)

        # Metric Comparison Bar
        _bar_data = {}
        for _lbl, _ser in [
            ("Dice", best_dice_series),
            ("IoU", best_iou_series),
            ("Precision", precision_series),
            ("Recall", recall_series),
            ("F1", f1_series),
        ]:
            if not _ser.empty:
                _bar_data[_lbl] = float(_ser.max())
        if _bar_data:
            render_section_header("📊 Metric Comparison")
            _bar_df = pd.DataFrame({"Metric": list(_bar_data.keys()), "Score": list(_bar_data.values())})
            st.bar_chart(_bar_df.set_index("Metric"))

    # Training Curves (interactive)
    render_section_header("📈 Training Curves")
    if not history.empty:
        _curve_tabs = st.tabs(["Loss Curve", "Dice Curve", "IoU Curve", "All Curves"])
        with _curve_tabs[0]:
            if {"Epoch", "Training Loss"}.issubset(history.columns):
                st.line_chart(history.set_index("Epoch")["Training Loss"])
            else:
                st.info("Training loss data is unavailable.")
        with _curve_tabs[1]:
            if {"Epoch", "Validation Dice"}.issubset(history.columns):
                st.line_chart(history.set_index("Epoch")["Validation Dice"])
            else:
                st.info("Validation Dice data is unavailable.")
        with _curve_tabs[2]:
            if {"Epoch", "Validation IoU"}.issubset(history.columns):
                st.line_chart(history.set_index("Epoch")["Validation IoU"])
            else:
                st.info("Validation IoU data is unavailable.")
        with _curve_tabs[3]:
            _all_cols = [c for c in ["Training Loss", "Validation Dice", "Validation IoU"] if c in history.columns]
            if "Epoch" in history.columns and _all_cols:
                st.line_chart(history.set_index("Epoch")[_all_cols])
            else:
                st.info("Combined curve data is unavailable.")
    else:
        # Fall back to saved plot images
        plot_paths = [
            ("Training Loss", PLOTS_DIR / "loss_curve.png"),
            ("Validation Dice", PLOTS_DIR / "dice_curve.png"),
            ("Validation IoU", PLOTS_DIR / "iou_curve.png"),
        ]
        plot_columns = st.columns(3)
        for column, (title, plot_path) in zip(plot_columns, plot_paths):
            with column:
                if plot_path.exists():
                    safe_image(str(plot_path), caption=title)
                else:
                    st.info(f"{title} plot is not available yet.")

    render_section_header("Confusion Matrix")
    info_tooltip(
        "eval_cm_info",
        "Confusion Matrix",
        "The confusion matrix shows four categories of pixel classification:<br><br>"
        "<strong>TP (True Positive)</strong> — Correctly predicted foreground pixels.<br>"
        "<strong>TN (True Negative)</strong> — Correctly predicted background pixels.<br>"
        "<strong>FP (False Positive)</strong> — Background pixels incorrectly predicted as foreground.<br>"
        "<strong>FN (False Negative)</strong> — Foreground pixels missed by the model.",
        "A confusion matrix is a simple grid that shows exactly where the AI got confused. "
        "<strong>True Positives</strong> and <strong>True Negatives</strong> are when the AI was completely correct. "
        "A <strong>False Positive</strong> is a false alarm—the AI thought there was a tumor, but it was just healthy tissue. "
        "A <strong>False Negative</strong> is the most dangerous error—there was a real tumor, but the AI completely missed it. "
        "In medical AI, we often tweak the system to minimize False Negatives, because missing a real tumor is much worse than a false alarm."
    )
    confusion_matrix_path = find_confusion_matrix_artifact()
    if confusion_matrix_path is None:
        st.info("No confusion matrix artifact was found.")
    elif confusion_matrix_path.suffix.lower() == ".csv":
        confusion_frame = load_csv_dataframe(confusion_matrix_path)
        if confusion_frame.empty:
            st.info("Confusion matrix CSV is empty.")
        else:
            st.dataframe(confusion_frame, use_container_width=True, hide_index=False)
    else:
        safe_image(str(confusion_matrix_path))

    render_section_header("📦 Other Artifacts")
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown('<div class="dashboard-card"><strong>Checkpoints</strong></div>', unsafe_allow_html=True)
        checkpoint_files = list_checkpoint_files()
        if checkpoint_files:
            for _ckpt in checkpoint_files:
                _sz = _ckpt.stat().st_size / 1e6
                st.markdown(
                    f'<div class="mini-card" style="margin-top:0.4rem;"><div style="font-weight:700;color:#0f172a;">{_ckpt.name}</div><div style="color:#64748b;font-size:0.82rem;">{_sz:.1f} MB</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No checkpoint files were found.")
    with col_right:
        st.markdown('<div class="dashboard-card"><strong>Prediction Folders</strong></div>', unsafe_allow_html=True)
        prediction_folders = list_prediction_directories()
        if prediction_folders:
            for _pf in prediction_folders:
                _count = sum(1 for p in _pf.iterdir() if p.is_file())
                st.markdown(
                    f'<div class="mini-card" style="margin-top:0.4rem;"><div style="font-weight:700;color:#0f172a;">{_pf.name}</div><div style="color:#64748b;font-size:0.82rem;">{_count} files</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No epoch prediction folders were found.")


def logo_image() -> Image.Image:
    """Return a logo image, creating a placeholder on demand if needed."""
    if LOGO_PATH.exists():
        return Image.open(LOGO_PATH).convert("RGBA")

    image = Image.new("RGBA", (512, 512), (15, 23, 42, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((36, 36, 476, 476), radius=48, fill=(30, 64, 175, 255))
    draw.ellipse((104, 104, 408, 408), fill=(14, 165, 233, 255))
    draw.text((178, 206), "MS", fill="white", font=ImageFont.load_default())
    return image


def encode_image(image: Image.Image) -> BytesIO:
    """Encode an image into memory for Streamlit download buttons."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def overlay_mask(image: Image.Image, mask: Image.Image, alpha: int = 120) -> Image.Image:
    """Create a simple overlay visualization for a predicted mask."""
    base = image.convert("RGBA").resize((image.width, image.height))
    mask_rgba = Image.new("RGBA", base.size, (255, 0, 0, 0))
    mask_resized = mask.convert("L").resize(base.size)
    mask_rgba.putalpha(mask_resized.point(lambda pixel: alpha if pixel > 0 else 0))
    return Image.alpha_composite(base, mask_rgba)


def binary_mask_to_pil(mask_array) -> Image.Image:
    """Convert a binary numpy mask to a grayscale PIL image."""
    array = (mask_array.astype("uint8") * 255).squeeze()
    return Image.fromarray(array, mode="L")
