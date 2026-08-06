"""Reusable dataset statistics utilities for medical segmentation datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}


@dataclass
class ResolutionSummary:
    """Store frequency information for image or mask resolutions."""

    counts: dict[str, int]

    def to_lines(self, title: str) -> list[str]:
        """Convert the resolution summary into readable text lines."""
        lines = [f"{title}:"]
        if not self.counts:
            lines.append("  None")
            return lines

        for resolution, count in sorted(self.counts.items()):
            lines.append(f"  {resolution} -> {count}")
        return lines


def is_supported_image(path: Path) -> bool:
    """Return ``True`` if a file uses one of the supported image formats."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def read_image_shape(path: Path) -> tuple[int, int] | None:
    """Read an image and return its ``(width, height)`` if it can be loaded."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    height, width = image.shape[:2]
    return width, height


def read_mask_array(path: Path) -> np.ndarray | None:
    """Load a mask as a grayscale numpy array."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask


def get_foreground_background_counts(mask: np.ndarray) -> tuple[int, int]:
    """Count foreground and background pixels in a segmentation mask."""
    foreground = int(np.count_nonzero(mask))
    total = int(mask.size)
    background = total - foreground
    return foreground, background


def get_foreground_background_percentages(
    mask: np.ndarray,
) -> tuple[float, float]:
    """Return the foreground and background percentages for a mask."""
    foreground, background = get_foreground_background_counts(mask)
    total = max(mask.size, 1)
    return (foreground / total) * 100.0, (background / total) * 100.0


def summarize_resolutions(resolutions: Iterable[tuple[int, int]]) -> ResolutionSummary:
    """Build a frequency table for image or mask resolutions."""
    counter = Counter(f"{width}x{height}" for width, height in resolutions)
    return ResolutionSummary(counts=dict(counter))


def average_dimensions(resolutions: Iterable[tuple[int, int]]) -> tuple[float, float]:
    """Compute the average width and height for a list of resolutions."""
    resolutions = list(resolutions)
    if not resolutions:
        return 0.0, 0.0

    widths = [width for width, _ in resolutions]
    heights = [height for _, height in resolutions]
    return float(sum(widths) / len(widths)), float(sum(heights) / len(heights))

