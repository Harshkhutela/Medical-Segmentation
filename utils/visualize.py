"""Visualization helpers for medical image segmentation predictions."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch


def _to_numpy(array_like):
    """Convert tensors to numpy arrays and leave numpy inputs unchanged."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _prepare_for_imshow(array_like, *, allow_rgb: bool = False):
    """Normalize common tensor/array shapes into Matplotlib-friendly layouts."""
    array = _to_numpy(array_like).squeeze()

    if array.ndim == 2:
        return array

    if allow_rgb and array.ndim == 3:
        if array.shape[-1] == 3:
            return array
        if array.shape[0] == 3:
            return np.transpose(array, (1, 2, 0))

    if array.ndim == 3 and array.shape[-1] == 1:
        return array.squeeze(-1)

    return np.squeeze(array)


def save_prediction_visualization(
    image,
    ground_truth_mask,
    predicted_mask,
    output_path,
):
    """Save a three-column comparison of an image and its segmentation masks.

    Args:
        image: RGB image array with shape ``(H, W, 3)``.
        ground_truth_mask: Binary ground-truth mask array with shape ``(H, W)``.
        predicted_mask: Binary prediction array with shape ``(H, W)``.
        output_path: Destination path for the generated PNG image.
    """
    # One row and exactly three columns make the result easy to compare.
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    image = _prepare_for_imshow(image, allow_rgb=True)
    ground_truth_mask = _prepare_for_imshow(ground_truth_mask)
    predicted_mask = _prepare_for_imshow(predicted_mask)

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ground_truth_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    # Request a non-blocking display so the command-line inference script ends.
    plt.show(block=False)
    plt.pause(0.001)
    plt.close(figure)
