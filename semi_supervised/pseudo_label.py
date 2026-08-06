"""Generate high-confidence pseudo masks for unlabeled medical images."""

import cv2
import torch

from active_learning.sampler import ActiveSampler
from configs.config import CHECKPOINT_PATH, DATASET_PATH, DEVICE


class PseudoLabelGenerator:
    """Create pseudo labels from confident predictions of the trained U-Net."""

    def __init__(self, confidence_threshold=0.90, checkpoint_file=None):
        """Set the confidence threshold and project-configured data locations.

        Args:
            confidence_threshold: Minimum image confidence required to save a
                predicted mask.
            checkpoint_file: Optional trained U-Net checkpoint path.
        """
        self.confidence_threshold = confidence_threshold
        self.checkpoint_file = (
            checkpoint_file
            if checkpoint_file is not None
            else CHECKPOINT_PATH / "best_model.pth"
        )
        self.pseudo_mask_directory = DATASET_PATH / "pseudo_masks"

        # ActiveSampler already finds unlabeled files and uses dataset preprocessing.
        self.sampler = ActiveSampler(checkpoint_file=self.checkpoint_file)

    def load_model(self):
        """Load the saved U-Net once before generating pseudo labels."""
        self.sampler.load_model()

    @staticmethod
    def calculate_confidence(probabilities):
        """Return mean per-pixel confidence across foreground and background.

        A pixel confidence is the probability of its most likely binary class.
        Averaging these values produces one confidence score for the image.
        """
        foreground_confidence = probabilities
        background_confidence = 1 - probabilities
        return torch.maximum(
            foreground_confidence,
            background_confidence,
        ).mean().item()

    def generate_pseudo_labels(self):
        """Generate masks for images meeting the configured confidence threshold.

        Returns:
            A dictionary with generated count, skipped count, and the average
            confidence of generated pseudo labels.
        """
        if self.sampler.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        image_paths = self.sampler.find_unlabeled_images()
        self.pseudo_mask_directory.mkdir(parents=True, exist_ok=True)

        generated = 0
        skipped = 0
        generated_confidences = []

        for image_path in image_paths:
            # Reuse the dataset loader's RGB conversion, resize, and normalization.
            image = self.sampler.dataset_loader._load_image(image_path)
            image_batch = image.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probabilities = torch.sigmoid(self.sampler.model(image_batch))

            confidence = self.calculate_confidence(probabilities)
            if confidence < self.confidence_threshold:
                skipped += 1
                continue

            # Threshold probabilities to create a 0/255 mask readable by OpenCV.
            binary_mask = (probabilities >= 0.5).squeeze().cpu().numpy()
            binary_mask = (binary_mask * 255).astype("uint8")
            output_file = self.pseudo_mask_directory / image_path.name
            if not cv2.imwrite(str(output_file), binary_mask):
                raise OSError(f"Unable to save pseudo mask: {output_file}")

            generated += 1
            generated_confidences.append(confidence)

        average_confidence = (
            sum(generated_confidences) / len(generated_confidences)
            if generated_confidences
            else 0.0
        )
        return {
            "generated": generated,
            "skipped": skipped,
            "average_confidence": average_confidence,
        }
