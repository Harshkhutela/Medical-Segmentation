"""Overlay helpers for XAI heatmaps and highlighted tumor regions."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap_image: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a heatmap with the original MRI image."""
    base = original_image.convert("RGBA")
    heatmap = heatmap_image.convert("RGBA").resize(base.size)
    heatmap.putalpha(int(255 * max(0.0, min(alpha, 1.0))))
    blended = Image.alpha_composite(base, heatmap)
    return blended.convert("RGB")


def _mask_bounds(mask_array: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bounding box for a binary mask."""
    coordinates = np.argwhere(mask_array > 0)
    if coordinates.size == 0:
        return None

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def create_attention_visualization(
    original_image: Image.Image,
    predicted_mask: Image.Image,
    heatmap_image: Image.Image,
    confidence: float = 0.0,
) -> Image.Image:
    """Create an attention visualization that highlights the tumor region."""
    base = overlay_heatmap_on_image(original_image, heatmap_image, alpha=0.5).convert("RGBA")
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()

    mask_array = np.asarray(predicted_mask.convert("L"))
    mask_bounds = _mask_bounds(mask_array)
    label_text = f"Prediction confidence: {confidence:.2%}"

    if mask_bounds is not None:
        x_min, y_min, x_max, y_max = mask_bounds
        padding = 6
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(base.size[0] - 1, x_max + padding)
        y_max = min(base.size[1] - 1, y_max + padding)

        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=(255, 215, 0, 255),
            width=4,
        )
        label_box = (x_min, max(0, y_min - 28), x_min + 230, y_min)
        draw.rectangle(label_box, fill=(255, 215, 0, 200))
        draw.text((label_box[0] + 6, label_box[1] + 6), "Highlighted tumor region", fill="black", font=font)
    else:
        draw.rectangle([(12, 12), (290, 52)], fill=(255, 215, 0, 200))
        draw.text((18, 18), "No confident tumor region", fill="black", font=font)

    draw.rectangle([(12, base.size[1] - 46), (290, base.size[1] - 10)], fill=(15, 23, 42, 180))
    draw.text((18, base.size[1] - 40), label_text, fill="white", font=font)
    return base.convert("RGB")
