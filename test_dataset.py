import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add the project root to Python's import path so the project modules can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config import IMAGE_FOLDER, MASK_FOLDER
from utils.dataset import MedicalSegmentationDataset


def main():
    """
    Load the dataset, print sample information, and display the first image and mask.

    This script is the first runnable milestone for the project. It gives a quick
    visual confirmation that the data pipeline is working.
    """
    dataset = MedicalSegmentationDataset()
    print(f"Total number of samples: {len(dataset)}")

    if len(dataset) == 0:
        print("\nNo image-mask pairs were found.")
        print(f"Please place images in: {IMAGE_FOLDER}")
        print(f"Please place matching masks in: {MASK_FOLDER}")
        print("Make sure each image and mask share the same filename stem, for example: sample1.png and sample1.png")
        return

    image_tensor, mask_tensor = dataset[0]

    print("First sample loaded successfully.")
    print("Image tensor shape:", tuple(image_tensor.shape))
    print("Mask tensor shape:", tuple(mask_tensor.shape))
    print("Image tensor dtype:", image_tensor.dtype)
    print("Mask tensor dtype:", mask_tensor.dtype)

    # Convert the tensors back to numpy arrays for visualization.
    image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_numpy = (image_numpy * 255.0).astype("uint8")

    mask_numpy = mask_tensor.cpu().numpy().astype("uint8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_numpy)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_numpy, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
