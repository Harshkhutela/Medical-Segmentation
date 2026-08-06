"""Dataset configuration helpers for small and large medical datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from configs.config import BASE_DIR, BATCH_SIZE, CHANNELS, IMAGE_SIZE, OUTPUT_PATH


DATASET_VARIANTS = {"BrainMRI", "BUSI", "ISIC", "Custom"}


def _get_env_path(name: str, default: Path) -> Path:
    """Read a path-like environment variable with a safe default."""
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return Path(raw_value).expanduser()


def _get_env_float(name: str, default: float) -> float:
    """Read a float environment variable with fallback support."""
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable with fallback support."""
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with fallback support."""
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class DatasetConfiguration:
    """Runtime configuration for dataset selection and large-scale training."""

    dataset_root: Path
    dataset_name: str = "Custom"
    image_size: int = IMAGE_SIZE
    channels: int = CHANNELS
    batch_size: int = BATCH_SIZE
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    num_workers: int = 4
    cache_images: bool = False
    pin_memory: bool = True
    seed: int = 42
    auto_split: bool = True
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    resume_training: bool = True
    output_dir: Path = OUTPUT_PATH

    @property
    def source_images_dir(self) -> Path:
        """Return the original image folder for the selected dataset."""
        return self.dataset_root / "images"

    @property
    def source_masks_dir(self) -> Path:
        """Return the original mask folder for the selected dataset."""
        return self.dataset_root / "masks"

    @property
    def train_images_dir(self) -> Path:
        """Return the training images folder created for large datasets."""
        return self.dataset_root / "train" / "images"

    @property
    def train_masks_dir(self) -> Path:
        """Return the training masks folder created for large datasets."""
        return self.dataset_root / "train" / "masks"

    @property
    def val_images_dir(self) -> Path:
        """Return the validation images folder created for large datasets."""
        return self.dataset_root / "val" / "images"

    @property
    def val_masks_dir(self) -> Path:
        """Return the validation masks folder created for large datasets."""
        return self.dataset_root / "val" / "masks"

    @classmethod
    def from_env(cls) -> "DatasetConfiguration":
        """Create a dataset configuration from environment variables."""
        base_root = _get_env_path("MEDSEG_DATASET_ROOT", BASE_DIR / "dataset")
        dataset_name = os.getenv("MEDSEG_DATASET_NAME", "Custom").strip() or "Custom"
        if dataset_name not in DATASET_VARIANTS:
            dataset_name = "Custom"

        dataset_root = base_root
        named_dataset = base_root / dataset_name
        if dataset_name != "Custom" and named_dataset.exists():
            dataset_root = named_dataset

        train_ratio = _get_env_float("MEDSEG_TRAIN_RATIO", 0.8)
        val_ratio = _get_env_float("MEDSEG_VAL_RATIO", 0.2)
        if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio > 1.0:
            train_ratio, val_ratio = 0.8, 0.2

        num_workers = _get_env_int("MEDSEG_NUM_WORKERS", max((os.cpu_count() or 2) - 1, 1))

        return cls(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            image_size=_get_env_int("MEDSEG_IMAGE_SIZE", IMAGE_SIZE),
            channels=_get_env_int("MEDSEG_CHANNELS", CHANNELS),
            batch_size=_get_env_int("MEDSEG_BATCH_SIZE", BATCH_SIZE),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            num_workers=max(num_workers, 0),
            cache_images=_get_env_bool("MEDSEG_CACHE_IMAGES", False),
            pin_memory=_get_env_bool("MEDSEG_PIN_MEMORY", True),
            seed=_get_env_int("MEDSEG_SPLIT_SEED", 42),
            auto_split=_get_env_bool("MEDSEG_AUTO_SPLIT", True),
            early_stopping_patience=_get_env_int("MEDSEG_EARLY_STOPPING", 5),
            scheduler_factor=_get_env_float("MEDSEG_SCHEDULER_FACTOR", 0.5),
            scheduler_patience=_get_env_int("MEDSEG_SCHEDULER_PATIENCE", 2),
            resume_training=_get_env_bool("MEDSEG_RESUME_TRAINING", True),
        )


def get_dataset_configuration() -> DatasetConfiguration:
    """Return the active dataset configuration."""
    return DatasetConfiguration.from_env()


def describe_dataset_configuration(config: DatasetConfiguration | None = None) -> str:
    """Create a readable summary of the selected dataset configuration."""
    config = config or get_dataset_configuration()
    return (
        f"Dataset root: {config.dataset_root}\n"
        f"Dataset name: {config.dataset_name}\n"
        f"Train ratio: {config.train_ratio:.2f}\n"
        f"Validation ratio: {config.val_ratio:.2f}\n"
        f"Workers: {config.num_workers}\n"
        f"Cache images: {config.cache_images}\n"
        f"Auto split: {config.auto_split}\n"
    )
