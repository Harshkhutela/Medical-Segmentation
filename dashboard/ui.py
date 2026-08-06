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
DEVELOPER_NAME = "Bhumi"
DEVELOPER_ROLE = "Research / ML Engineering"
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
            :root {
                --bg-start: #f5fbff;
                --bg-end: #eef4ff;
                --card-border: rgba(15, 23, 42, 0.08);
                --card-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
                --accent: #0f766e;
                --accent-2: #2563eb;
                --text-main: #0f172a;
                --text-muted: #475569;
            }

            .stApp {
                background: linear-gradient(180deg, var(--bg-start), var(--bg-end));
                color: var(--text-main);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1.6rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
                color: white;
            }

            [data-testid="stSidebar"] * {
                color: white;
            }

            .dashboard-shell {
                max-width: 1320px;
                margin: 0 auto;
            }

            .hero-card {
                padding: 2rem;
                border-radius: 28px;
                background:
                    radial-gradient(circle at top right, rgba(37, 99, 235, 0.16), transparent 30%),
                    linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 249, 255, 0.92));
                border: 1px solid var(--card-border);
                box-shadow: var(--card-shadow);
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: rgba(15, 118, 110, 0.08);
                color: var(--accent);
                font-weight: 700;
                letter-spacing: 0.02em;
                margin-bottom: 1rem;
            }

            .section-title {
                font-size: 1.3rem;
                font-weight: 800;
                color: var(--text-main);
                margin: 1.4rem 0 0.35rem;
            }

            .section-subtitle {
                color: var(--text-muted);
                margin-bottom: 0.6rem;
            }

            .dashboard-card {
                padding: 1rem 1rem 0.95rem;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid var(--card-border);
                box-shadow: var(--card-shadow);
            }

            .mini-card {
                padding: 1rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,250,255,0.95));
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            }

            .pipeline-step {
                padding: 1rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.9);
                border: 1px solid rgba(148,163,184,0.16);
                text-align: center;
                min-height: 130px;
            }

            .pipeline-arrow {
                font-size: 2rem;
                color: #0f766e;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }

            .footer-bar {
                margin-top: 2rem;
                padding: 0.8rem 1rem;
                border-top: 1px solid rgba(148, 163, 184, 0.2);
                color: var(--text-muted);
                font-size: 0.92rem;
            }

            .sidebar-title {
                margin-top: 0.75rem;
                font-size: 1.35rem;
                font-weight: 800;
                line-height: 1.15;
            }

            .sidebar-chip {
                display: inline-block;
                padding: 0.28rem 0.6rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.16);
                margin-right: 0.4rem;
                margin-bottom: 0.35rem;
                font-size: 0.82rem;
            }

            .stMetric {
                background: rgba(255,255,255,0.84);
                border: 1px solid rgba(148,163,184,0.14);
                border-radius: 18px;
                padding: 0.6rem 0.85rem;
                box-shadow: var(--card-shadow);
            }

            div[data-testid="stFileUploader"] {
                background: rgba(255,255,255,0.75);
                border-radius: 18px;
                border: 1px dashed rgba(15,118,110,0.35);
                padding: 0.35rem;
            }

            [data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148,163,184,0.12);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Render the professional branded sidebar."""
    st.image(logo_image(), use_container_width=True)
    st.markdown(
        f"""
        <div class="sidebar-title">{APP_TITLE}</div>
        <div style="margin-top:0.35rem; opacity:0.9;">{PROJECT_TITLE}</div>
        <div style="margin-top:0.8rem;">
            <span class="sidebar-chip">Version {APP_VERSION}</span>
            <span class="sidebar-chip">{LICENSE_NAME}</span>
        </div>
        <div style="margin-top:0.9rem; line-height:1.5;">
            <strong>Developer</strong><br />
            {DEVELOPER_NAME}<br />
            <span style="opacity:0.85;">{DEVELOPER_ROLE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render a small footer with project metadata."""
    st.markdown(
        f"""
        <div class="footer-bar">
            <strong>GitHub:</strong> {GITHUB_PLACEHOLDER}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>License:</strong> {LICENSE_NAME}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Project Version:</strong> {APP_VERSION}
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
    dataset_config = get_dataset_configuration()
    dataset_root = dataset_config.dataset_root
    labeled_images = count_supported_images(dataset_root / "images")
    unlabeled_images = count_supported_images(dataset_root / "unlabeled")
    pseudo_labels = count_supported_images(dataset_root / "pseudo_masks")
    train_images = count_supported_images(dataset_root / "train" / "images")
    val_images = count_supported_images(dataset_root / "val" / "images")

    history = load_history_dataframe()
    best_dice, best_iou = best_metric_values(history)

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">Medical AI Research Dashboard</div>
            <h1 style="margin:0 0 0.5rem 0; font-size:2.1rem;">
                {PROJECT_TITLE}
            </h1>
            <p style="font-size:1.04rem; color:#334155; max-width: 900px; line-height:1.7;">
                A polished research-grade interface for medical image
                segmentation experiments, active learning, semi-supervised
                learning, and evaluation. The dashboard is designed for
                internship demonstrations and reproducible research walkthroughs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header(
        "Project Overview",
        "The application wraps the existing U-Net workflow in a clean, research-focused interface.",
    )

    overview_left, overview_right = st.columns([1.2, 0.8], gap="large")
    with overview_left:
        st.markdown(
            """
            <div class="dashboard-card">
                <strong>What this project demonstrates</strong>
                <ul style="margin-top:0.7rem; line-height:1.8; color:#334155;">
                    <li>Supervised segmentation with a U-Net backbone.</li>
                    <li>Training history and experiment tracking.</li>
                    <li>Uncertainty-based active learning on unlabeled scans.</li>
                    <li>Pseudo-label generation for semi-supervised expansion.</li>
                    <li>Evaluation and visualization of segmentation results.</li>
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
                <p style="margin-top:0.6rem; color:#334155; line-height:1.7;">
                    The dashboard connects the full experimental loop from
                    dataset preparation to evaluation so each stage can be
                    explained clearly in a demo or viva.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_header("Workflow Diagram", "A concise visual of the end-to-end research loop.")
    workflow = st.columns([1.1, 0.2, 1.1, 0.2, 1.1, 0.2, 1.1, 0.2, 1.1, 0.2, 1.1])
    stages = [
        ("Dataset", "Pairs, splits, stats"),
        ("Training", "U-Net optimization"),
        ("Active Learning", "Uncertainty ranking"),
        ("Pseudo Labeling", "High-confidence masks"),
        ("Retraining", "Merged supervision"),
        ("Evaluation", "Metrics and plots"),
    ]
    for index, (title, subtitle) in enumerate(stages):
        with workflow[index * 2]:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div style="font-weight:800; font-size:1.03rem; color:#0f172a;">{title}</div>
                    <div style="margin-top:0.55rem; color:#475569; line-height:1.5;">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if index < len(stages) - 1:
            with workflow[index * 2 + 1]:
                st.markdown(
                    '<div class="pipeline-arrow">&rarr;</div>',
                    unsafe_allow_html=True,
                )

    render_section_header("Dataset Statistics", "Quick readout from the prepared project folders.")
    render_metric_cards(
        [
            ("Labeled Images", labeled_images),
            ("Train Images", train_images),
            ("Validation Images", val_images),
            ("Unlabeled Images", unlabeled_images),
            ("Pseudo Labels", pseudo_labels),
            ("Best Dice", "N/A" if history.empty else f"{best_dice:.4f}"),
            ("Best IoU", "N/A" if history.empty else f"{best_iou:.4f}"),
        ]
    )

    render_section_header("Model Information", "The project uses a compact U-Net for binary segmentation.")
    model_cards = st.columns(4)
    model_details = [
        ("Architecture", "U-Net"),
        ("Input Channels", "3"),
        ("Output Channels", "1"),
        ("Input Size", "256 x 256"),
    ]
    for column, (label, value) in zip(model_cards, model_details):
        with column:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div style="color:#64748b; font-size:0.86rem;">{label}</div>
                    <div style="font-size:1.2rem; font-weight:800; margin-top:0.25rem; color:#0f172a;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="dashboard-card" style="margin-top:1rem;">
            <strong>Why this matters</strong>
            <p style="margin-top:0.55rem; color:#334155; line-height:1.75;">
                This layout makes it easy to explain the dataset, model,
                training loop, active learning cycle, and evaluation results as
                a single research system rather than separate scripts.
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

    render_section_header("Experiment History")
    st.dataframe(history, use_container_width=True, hide_index=True)

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
            <div class="hero-badge">Active Learning</div>
            <h1 style="margin:0 0 0.4rem 0;">Uncertainty Ranking Dashboard</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Review uncertainty scores and inspect the most informative
                unlabeled samples for annotation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
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
        max_items = max(1, min(20, len(frame)))
        top_k = st.slider("Top samples to display", 1, max_items, min(5, max_items))

        frame = frame.copy()
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce").fillna(0)
        frame = frame.sort_values(by=score_column, ascending=False)

        render_section_header("Selection Summary")
        render_metric_cards(
            [
                ("Total Samples", len(frame)),
                ("Top-K", top_k),
            ]
        )

    with right:
        render_section_header("Top Uncertain Samples")
        top_samples = frame.head(top_k)[["filename", score_column]]
        for _, row in top_samples.iterrows():
            st.markdown(
                f"""
                <div class="mini-card" style="margin-bottom:0.7rem;">
                    <div style="font-weight:800; color:#0f172a;">{row['filename']}</div>
                    <div style="color:#475569; margin-top:0.25rem;">
                        {score_column.replace('_', ' ').title()}: {float(row[score_column]):.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_section_header("Uncertainty Table")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    render_section_header("Uncertainty Chart")
    st.bar_chart(frame.head(top_k).set_index("filename")[score_column])


def render_evaluation_page() -> None:
    """Render the experiment evaluation page."""
    frame = load_experiment_dataframe()
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Evaluation</div>
            <h1 style="margin:0 0 0.4rem 0;">Experiment Summary & Metrics</h1>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Review the final metrics, publication-style plots, and available
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
        metrics = st.columns(6)
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
        for column, (label, value) in zip(metrics, summary_values):
            with column:
                if value is None or pd.isna(value):
                    display_value = "N/A"
                elif "Epoch" in label:
                    display_value = int(value)
                else:
                    display_value = f"{float(value):.4f}"
                st.metric(label, display_value)

        render_section_header("Experiment Results")
        st.dataframe(frame, use_container_width=True, hide_index=True)

    render_section_header("Training Curves")
    plot_paths = [
        ("Training Loss", PLOTS_DIR / "loss_curve.png"),
        ("Validation Dice", PLOTS_DIR / "dice_curve.png"),
        ("Validation IoU", PLOTS_DIR / "iou_curve.png"),
    ]
    plot_columns = st.columns(3)
    for column, (title, plot_path) in zip(plot_columns, plot_paths):
        with column:
            if plot_path.exists():
                st.image(str(plot_path), caption=title, use_container_width=True)
            else:
                st.info(f"{title} plot is not available yet.")

    render_section_header("Confusion Matrix")
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
        st.image(str(confusion_matrix_path), use_container_width=True)

    render_section_header("Other Artifacts")
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        checkpoint_files = list_checkpoint_files()
        if checkpoint_files:
            st.write([str(path) for path in checkpoint_files])
        else:
            st.info("No checkpoint files were found.")
    with col_right:
        prediction_folders = list_prediction_directories()
        if prediction_folders:
            st.write([str(path) for path in prediction_folders])
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


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Create a simple overlay visualization for a predicted mask."""
    base = image.convert("RGBA").resize((image.width, image.height))
    mask_rgba = Image.new("RGBA", base.size, (255, 0, 0, 0))
    mask_resized = mask.convert("L").resize(base.size)
    mask_rgba.putalpha(mask_resized.point(lambda pixel: 120 if pixel > 0 else 0))
    return Image.alpha_composite(base, mask_rgba)


def binary_mask_to_pil(mask_array) -> Image.Image:
    """Convert a binary numpy mask to a grayscale PIL image."""
    array = (mask_array.astype("uint8") * 255).squeeze()
    return Image.fromarray(array, mode="L")
