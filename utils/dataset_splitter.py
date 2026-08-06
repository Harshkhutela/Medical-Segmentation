"""Large-dataset helpers for split creation and efficient data access."""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Iterable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from configs.dataset_config import DatasetConfiguration, get_dataset_configuration
from utils.dataset import MedicalSegmentationDataset


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}


@dataclass(frozen=True)
class SplitPaths:
    """Store the folder layout for train and validation data."""

    train_images: Path
    train_masks: Path
    val_images: Path
    val_masks: Path


class DatasetSplitManager:
    """Create a train/validation split without changing the training pipeline."""

    def __init__(self, config: DatasetConfiguration | None = None):
        """Store the runtime configuration and derived paths."""
        self.config = config or get_dataset_configuration()
        self.dataset_root = self.config.dataset_root
        self.source_images = self.config.source_images_dir
        self.source_masks = self.config.source_masks_dir
        self.train_images = self.config.train_images_dir
        self.train_masks = self.config.train_masks_dir
        self.val_images = self.config.val_images_dir
        self.val_masks = self.config.val_masks_dir

    def ensure_split_layout(self) -> SplitPaths:
        """Reuse a valid split or rebuild it when required."""
        if self._existing_split_is_valid():
            print("Using existing dataset split.")
            return SplitPaths(
                train_images=self.train_images,
                train_masks=self.train_masks,
                val_images=self.val_images,
                val_masks=self.val_masks,
            )

        self._clear_existing_split()

        pairs = self._discover_pairs()
        if not pairs:
            raise ValueError(
                "No valid image-mask pairs were found for automatic splitting."
            )

        train_pairs, val_pairs = train_test_split(
            pairs,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            shuffle=True,
        )
        self._materialize_split(self.train_images, self.train_masks, train_pairs)
        self._materialize_split(self.val_images, self.val_masks, val_pairs)
        self._write_manifest(self.dataset_root / "train_manifest.csv", train_pairs)
        self._write_manifest(self.dataset_root / "val_manifest.csv", val_pairs)

        train_images_count = self._count_files(self.train_images)
        train_masks_count = self._count_files(self.train_masks)
        val_images_count = self._count_files(self.val_images)
        val_masks_count = self._count_files(self.val_masks)

        print(f"Total Images : {len(pairs)}")
        print(f"Train Images : {train_images_count}")
        print(f"Validation Images : {val_images_count}")

        if train_images_count != train_masks_count:
            raise RuntimeError(
                "Train split verification failed: image and mask counts do not match."
            )
        if val_images_count != val_masks_count:
            raise RuntimeError(
                "Validation split verification failed: image and mask counts do not match."
            )

        return SplitPaths(
            train_images=self.train_images,
            train_masks=self.train_masks,
            val_images=self.val_images,
            val_masks=self.val_masks,
        )

    def _existing_split_is_valid(self) -> bool:
        """Check whether the on-disk split is already usable."""
        return all(
            (
                self._split_folder_is_valid(self.train_images, self.train_masks),
                self._split_folder_is_valid(self.val_images, self.val_masks),
            )
        )

    def _split_folder_is_valid(self, images_dir: Path, masks_dir: Path) -> bool:
        """Validate a single split folder pair."""
        if not images_dir.is_dir() or not masks_dir.is_dir():
            return False

        image_count = self._count_files(images_dir)
        mask_count = self._count_files(masks_dir)
        if image_count == 0 or mask_count == 0:
            return False
        if image_count != mask_count:
            return False

        image_keys = self._split_file_keys(images_dir)
        mask_keys = self._split_file_keys(masks_dir)
        return image_keys == mask_keys

    def _clear_existing_split(self) -> None:
        """Remove any previous split folders before creating a new one."""
        for folder in (
            self.train_images.parent,
            self.val_images.parent,
        ):
            if folder.is_dir():
                shutil.rmtree(folder)

        for manifest_name in ("train_manifest.csv", "val_manifest.csv"):
            manifest_path = self.dataset_root / manifest_name
            if manifest_path.exists():
                manifest_path.unlink()

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        """Match source images and masks by filename stem recursively."""
        if not self.source_images.is_dir() or not self.source_masks.is_dir():
            return []

        mask_lookup: dict[str, Path] = {}
        for path in self.source_masks.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                for key in self._candidate_keys(path, self.source_masks):
                    mask_lookup.setdefault(key, path)

        pairs: list[tuple[Path, Path]] = []
        used_masks: set[Path] = set()
        for image_path in sorted(self.source_images.rglob("*")):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS
            ):
                continue
            mask_path = self._find_matching_mask(
                image_path,
                mask_lookup,
                used_masks,
            )
            if mask_path is not None:
                pairs.append((image_path, mask_path))
                used_masks.add(mask_path)
        return pairs

    def _candidate_keys(self, path: Path, root: Path) -> list[str]:
        """Generate matching keys for an image or mask path."""
        relative_stem = path.relative_to(root).with_suffix("").as_posix().lower()
        stem = path.stem.lower()
        base_stem = self._normalize_key(stem)
        relative_base = self._normalize_key(relative_stem)
        return [
            relative_stem,
            relative_base,
            stem,
            base_stem,
        ]

    def _find_matching_mask(
        self,
        image_path: Path,
        mask_lookup: dict[str, Path],
        used_masks: set[Path],
    ) -> Path | None:
        """Find the best matching mask for a source image."""
        relative_stem = image_path.relative_to(self.source_images).with_suffix("").as_posix().lower()
        candidates = [
            relative_stem,
            self._normalize_key(relative_stem),
            image_path.stem.lower(),
            self._normalize_key(image_path.stem.lower()),
        ]

        for key in candidates:
            mask_path = mask_lookup.get(key)
            if mask_path is not None and mask_path not in used_masks:
                return mask_path
        return None

    def _normalize_key(self, value: str) -> str:
        """Normalize common image and mask naming patterns for pairing."""
        normalized = value.replace("\\", "/").rsplit("/", 1)[-1]
        prefix_patterns = ("image_", "img_", "mask_", "msk_")
        suffix_patterns = ("_mask", "-mask", "_label", "-label", "_seg", "-seg")

        for prefix in prefix_patterns:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        for suffix in suffix_patterns:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

        return normalized

    def _count_files(self, folder: Path) -> int:
        """Count supported files in a split directory recursively."""
        if not folder.exists():
            return 0
        return sum(
            1
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _split_file_keys(self, folder: Path) -> list[str]:
        """Return normalized file keys for a split folder."""
        keys: list[str] = []
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                keys.append(self._normalize_key(path.stem.lower()))
        return sorted(keys)

    def _materialize_split(
        self,
        images_dir: Path,
        masks_dir: Path,
        pairs: Iterable[tuple[Path, Path]],
    ) -> None:
        """Create split directories using hardlinks when possible."""
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in pairs:
            self._link_or_copy(image_path, images_dir / image_path.name)
            self._link_or_copy(mask_path, masks_dir / mask_path.name)

    def _link_or_copy(self, source: Path, destination: Path) -> None:
        """Prefer hardlinks for efficiency and fall back to regular copies."""
        if destination.exists():
            return

        try:
            os.link(source, destination)
        except OSError:
            copy2(source, destination)

    def _write_manifest(
        self,
        manifest_path: Path,
        pairs: Iterable[tuple[Path, Path]],
    ) -> None:
        """Write a lightweight CSV manifest for traceability."""
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_name",
                    "mask_name",
                    "image_path",
                    "mask_path",
                ],
            )
            writer.writeheader()
            for image_path, mask_path in pairs:
                writer.writerow(
                    {
                        "image_name": image_path.name,
                        "mask_name": mask_path.name,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                    }
                )


class CachedSegmentationDataset(Dataset):
    """Wrap the existing dataset loader with optional tensor caching."""

    def __init__(
        self,
        image_folder: str | Path,
        mask_folder: str | Path,
        image_size: int | None = None,
        cache_images: bool = False,
        cache_limit: int | None = None,
    ) -> None:
        """Create a wrapper around the existing medical segmentation dataset."""
        self.base_dataset = MedicalSegmentationDataset(
            image_folder=image_folder,
            mask_folder=mask_folder,
            image_size=image_size,
        )
        self.cache_images = cache_images
        self.cache_limit = cache_limit
        self._cache: dict[int, tuple] = {}
        self._cache_order: list[int] = []

    def __len__(self) -> int:
        """Return the number of paired samples."""
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        """Load a sample lazily and cache it if requested."""
        if self.cache_images and index in self._cache:
            return self._cache[index]

        sample = self.base_dataset[index]
        if self.cache_images:
            self._store_in_cache(index, sample)
        return sample

    def _store_in_cache(self, index: int, sample) -> None:
        """Store a sample in memory using a simple FIFO eviction rule."""
        if self.cache_limit is not None and len(self._cache) >= self.cache_limit:
            oldest_index = self._cache_order.pop(0)
            self._cache.pop(oldest_index, None)

        self._cache[index] = sample
        self._cache_order.append(index)
