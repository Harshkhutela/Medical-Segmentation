import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from configs.config import IMAGE_FOLDER, IMAGE_SIZE, MASK_FOLDER


class MedicalSegmentationDataset(Dataset):
    """
    Simple PyTorch dataset for loading paired medical images and masks.

    The loader reads images from the dataset/images folder and masks from
    dataset/masks. It matches them by filename stem (for example, sample1.png
    and sample1.png, or sample1.jpg and sample1.png).
    """

    def __init__(self, image_folder=None, mask_folder=None, image_size=None):
        """
        Initialize the dataset.

        Args:
            image_folder: Optional path to the image directory.
            mask_folder: Optional path to the mask directory.
            image_size: Output size for both image and mask.
        """
        self.image_folder = Path(image_folder) if image_folder is not None else IMAGE_FOLDER
        self.mask_folder = Path(mask_folder) if mask_folder is not None else MASK_FOLDER
        self.image_size = image_size if image_size is not None else IMAGE_SIZE

        # Build the list of matching image-mask pairs once at initialization.
        self.samples = self._build_samples()

    def _build_samples(self):
        """
        Create a list of paired image and mask file paths.

        The function looks for image files in the image folder and tries to find
        a mask with the same filename stem in the mask folder.
        """
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        image_paths = []
        if self.image_folder.exists():
            image_paths = sorted(
                path for path in self.image_folder.iterdir()
                if path.is_file() and path.suffix.lower() in supported_extensions
            )

        mask_paths = []
        if self.mask_folder.exists():
            mask_paths = sorted(
                path for path in self.mask_folder.iterdir()
                if path.is_file() and path.suffix.lower() in supported_extensions
            )

        # Use normalized filename stems so common naming schemes can be paired
        # correctly, while still supporting exact stem matches.
        mask_lookup = {}
        for path in mask_paths:
            raw_key = path.stem
            normalized_key = self._normalize_stem(path.stem)
            mask_lookup.setdefault(raw_key, path)
            mask_lookup.setdefault(normalized_key, path)

        samples = []
        for image_path in image_paths:
            image_key = image_path.stem
            normalized_image_key = self._normalize_stem(image_path.stem)

            mask_path = mask_lookup.get(image_key)
            if mask_path is None:
                mask_path = mask_lookup.get(normalized_image_key)

            if mask_path is not None:
                samples.append((image_path, mask_path))

        return samples

    def _normalize_stem(self, filename_stem):
        """
        Normalize common image and mask naming patterns before comparison.

        This keeps backward compatibility with datasets that already use the
        same name for image and mask, while also supporting names like:

        - image_000001.png  <->  mask_000001.png
        - img_000001.png    <->  msk_000001.png
        - lesion_01_mask.png <-> lesion_01.png
        """
        normalized = filename_stem.lower()

        prefix_patterns = ("image_", "img_", "mask_", "msk_")
        suffix_patterns = ("_mask", "-mask", "_label", "-label", "_seg", "-seg")

        for prefix in prefix_patterns:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break

        for suffix in suffix_patterns:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break

        return normalized

    def __len__(self):
        """Return the total number of available image-mask pairs."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load one image and its corresponding mask.

        Returns:
            image_tensor: RGB image as a torch.Tensor of shape (3, H, W)
            mask_tensor: Grayscale mask as a torch.Tensor of shape (H, W)
        """
        if index >= len(self.samples):
            raise IndexError("Dataset index is out of range")

        image_path, mask_path = self.samples[index]
        image_tensor = self._load_image(image_path)
        mask_tensor = self._load_mask(mask_path)
        return image_tensor, mask_tensor

    def _load_image(self, image_path):
        """
        Read an image from disk, resize it, normalize it, and convert it to a tensor.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        # OpenCV loads images in BGR order. Convert to RGB so matplotlib and PyTorch
        # display the channels correctly.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to the configured spatial size.
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Convert to a floating-point tensor and normalize to [0, 1].
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        return image_tensor

    def _load_mask(self, mask_path):
        """
        Read a segmentation mask, resize it, convert it to grayscale, and convert it to a tensor.
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")

        # Use nearest-neighbor interpolation for masks so label values remain intact.
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Convert to a float tensor with shape (H, W).
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return mask_tensor
