"""Smoke test for the U-Net medical image segmentation model."""

import torch

from configs.config import DEVICE, IMAGE_SIZE
from models import UNet


def count_trainable_parameters(model):
    """Return the number of model parameters that will be optimized."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def main():
    """Build the model, run an RGB test image, and verify its output shape."""
    print("Medical Segmentation Project\n")
    print("Creating U-Net...\n")

    model = UNet().to(DEVICE)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(output.shape) == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(output.shape)}."
    )

    print(f"Input Shape :\n{dummy_input.shape}\n")
    print(f"Output Shape :\n{output.shape}\n")
    print(f"Trainable Parameters :\n{count_trainable_parameters(model)}\n")
    print(f"Device :\n{DEVICE}\n")
    print("Model created successfully.")


if __name__ == "__main__":
    main()
