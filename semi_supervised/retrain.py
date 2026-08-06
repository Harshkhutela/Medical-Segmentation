"""Prepare a separate paired dataset for semi-supervised retraining."""

from shutil import copy2

from configs.config import DATASET_PATH, IMAGE_FOLDER, MASK_FOLDER
from utils.dataset import MedicalSegmentationDataset


def _copy_pairs(dataset, destination_images, destination_masks):
    """Copy paired samples to a destination without modifying their sources."""
    copied_samples = 0
    for image_path, mask_path in dataset.samples:
        copy2(image_path, destination_images / image_path.name)
        copy2(mask_path, destination_masks / mask_path.name)
        copied_samples += 1
    return copied_samples


def prepare_retraining_dataset():
    """Create a combined labeled and pseudo-labeled dataset for retraining.

    Original image-mask pairs and unlabeled image-pseudo-mask pairs are copied
    into ``dataset/retraining``. The source datasets are never moved or edited.

    Returns:
        A dictionary containing original, pseudo, and final paired sample counts.
    """
    unlabeled_directory = DATASET_PATH / "unlabeled"
    pseudo_mask_directory = DATASET_PATH / "pseudo_masks"
    retraining_directory = DATASET_PATH / "retraining"
    destination_images = retraining_directory / "images"
    destination_masks = retraining_directory / "masks"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_masks.mkdir(parents=True, exist_ok=True)

    # The existing loader guarantees only properly matched source pairs are copied.
    original_dataset = MedicalSegmentationDataset(
        image_folder=IMAGE_FOLDER,
        mask_folder=MASK_FOLDER,
    )
    pseudo_dataset = MedicalSegmentationDataset(
        image_folder=unlabeled_directory,
        mask_folder=pseudo_mask_directory,
    )

    original_count = _copy_pairs(
        original_dataset,
        destination_images,
        destination_masks,
    )
    pseudo_count = _copy_pairs(
        pseudo_dataset,
        destination_images,
        destination_masks,
    )
    return {
        "original_count": original_count,
        "pseudo_count": pseudo_count,
        "final_count": original_count + pseudo_count,
        "dataset_path": retraining_directory,
    }
