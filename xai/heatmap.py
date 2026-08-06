"""Heatmap rendering helpers for XAI visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def normalize_heatmap(heatmap: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize a heatmap to the [0, 1] range."""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    array = np.asarray(heatmap, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(array.min()) if array.size else 0.0
    maximum = float(array.max()) if array.size else 0.0
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum + 1e-8)


def heatmap_to_image(
    heatmap: np.ndarray | torch.Tensor,
    target_size: tuple[int, int] | None = None,
    colormap: str = "jet",
) -> Image.Image:
    """Convert a normalized heatmap into a colorful RGB image."""
    normalized = normalize_heatmap(heatmap)
    colored = plt.get_cmap(colormap)(normalized)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    image = Image.fromarray(colored, mode="RGB")
    if target_size is not None:
        image = image.resize(target_size, Image.BILINEAR)
    return image


def save_heatmap_image(
    heatmap_image: Image.Image | np.ndarray | torch.Tensor,
    output_path: Path,
) -> None:
    """Save a heatmap artifact to disk."""
    if not isinstance(heatmap_image, Image.Image):
        heatmap_image = heatmap_to_image(heatmap_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_image.save(output_path)
