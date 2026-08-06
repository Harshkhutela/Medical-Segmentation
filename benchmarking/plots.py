"""Plotting helpers for benchmark comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    ("dice", "Dice"),
    ("iou", "IoU"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]


def save_benchmark_artifacts(frame: pd.DataFrame, output_dir: Path) -> None:
    """Save bar charts and a comparison table for benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for column, title in METRIC_COLUMNS:
        if column not in frame.columns:
            continue
        _save_metric_bar_chart(frame, column, title, output_dir / f"{column}_comparison.png")

    _save_comparison_table(frame, output_dir / "benchmark_table.png")


def _save_metric_bar_chart(frame: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    """Save a single bar chart for one benchmark metric."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(frame["model"].astype(str), pd.to_numeric(frame[column], errors="coerce").fillna(0))
    axis.set_title(f"{title} Comparison")
    axis.set_ylabel(title)
    axis.set_xlabel("Model")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_comparison_table(frame: pd.DataFrame, output_path: Path) -> None:
    """Save a comparison table as a Matplotlib figure."""
    display_frame = frame.copy()
    for column, _ in METRIC_COLUMNS:
        if column in display_frame.columns:
            display_frame[column] = pd.to_numeric(display_frame[column], errors="coerce").fillna(0).map(
                lambda value: f"{value:.4f}"
            )

    figure, axis = plt.subplots(figsize=(10, max(2.5, 0.6 * len(display_frame) + 1)))
    axis.axis("off")
    table = axis.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
