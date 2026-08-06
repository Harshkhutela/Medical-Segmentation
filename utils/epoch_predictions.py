"""Utilities for saving prediction examples after each training epoch."""

from __future__ import annotations

from pathlib import Path

import torch

from configs.config import DEVICE
from utils.visualize import save_prediction_visualization


def _tensor_to_numpy_image(image_tensor: torch.Tensor):
    """Convert a channel-first tensor into a displayable RGB image."""
    return image_tensor.detach().cpu().permute(1, 2, 0).numpy()


def _tensor_to_numpy_mask(mask_tensor: torch.Tensor):
    """Convert a tensor mask into a binary numpy array."""
    return (mask_tensor.detach().cpu().numpy() > 0).astype("uint8")


def save_epoch_prediction_examples(
    model: torch.nn.Module,
    data_loader,
    epoch: int,
    output_root: str | Path,
    max_samples: int = 3,
) -> list[Path]:
    """Save a few prediction examples for the given epoch."""
    output_root = Path(output_root)
    epoch_dir = output_root / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    model.eval()

    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(data_loader):
            if len(saved_paths) >= max_samples:
                break

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = images.size(0)
            for sample_index in range(batch_size):
                if len(saved_paths) >= max_samples:
                    break

                image = _tensor_to_numpy_image(images[sample_index])
                mask = _tensor_to_numpy_mask(masks[sample_index])
                prediction = _tensor_to_numpy_mask(predictions[sample_index])

                output_file = epoch_dir / f"sample_{len(saved_paths) + 1:02d}.png"
                save_prediction_visualization(
                    image,
                    mask,
                    prediction,
                    output_file,
                )
                saved_paths.append(output_file)

    return saved_paths

