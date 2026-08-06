"""Visualization helpers for medical image segmentation predictions."""

import matplotlib.pyplot as plt


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
