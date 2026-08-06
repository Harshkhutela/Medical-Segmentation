"""Model-based uncertainty scoring for unlabeled medical images."""

from pathlib import Path

import torch

from active_learning.uncertainty import entropy, least_confidence, margin_sampling
from configs.config import CHECKPOINT_PATH, DATASET_PATH, DEVICE
from inference import load_model
from utils.dataset import MedicalSegmentationDataset


class ActiveSampler:
    """Score unlabeled images and rank them by prediction uncertainty."""

    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, checkpoint_file=None, unlabeled_directory=None):
        """Initialize the sampler with project-configured model and data paths.

        Args:
            checkpoint_file: Optional trained model path.
            unlabeled_directory: Optional directory containing images to score.
        """
        self.checkpoint_file = (
            Path(checkpoint_file)
            if checkpoint_file is not None
            else CHECKPOINT_PATH / "best_model.pth"
        )
        self.unlabeled_directory = (
            Path(unlabeled_directory)
            if unlabeled_directory is not None
            else DATASET_PATH / "unlabeled"
        )
        self.model = None

        # Reuse the existing dataset transform to keep preprocessing consistent.
        self.dataset_loader = MedicalSegmentationDataset()

    def load_model(self):
        """Load the trained U-Net once before scoring unlabeled images."""
        self.model = load_model(self.checkpoint_file)

    def find_unlabeled_images(self):
        """Return sorted supported image files from the unlabeled directory."""
        if not self.unlabeled_directory.is_dir():
            return []

        return sorted(
            path for path in self.unlabeled_directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        )

    def score_image(self, image_path):
        """Run one image through U-Net and return its uncertainty measurements."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # _load_image is the existing loader's RGB resize and normalization step.
        image = self.dataset_loader._load_image(image_path)
        image_batch = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(image_batch))

        least_confidence_score = least_confidence(probabilities)
        entropy_score = entropy(probabilities)
        margin_score = margin_sampling(probabilities)
        final_score = (
            least_confidence_score + entropy_score + margin_score
        ) / 3

        return {
            "filename": image_path.name,
            "least_confidence": least_confidence_score,
            "entropy": entropy_score,
            "margin": margin_score,
            "final_score": final_score,
        }

    def score_unlabeled_images(self):
        """Score every unlabeled image and return records sorted most uncertain first."""
        image_paths = self.find_unlabeled_images()
        scores = [self.score_image(image_path) for image_path in image_paths]
        return sorted(scores, key=lambda record: record["final_score"], reverse=True)
