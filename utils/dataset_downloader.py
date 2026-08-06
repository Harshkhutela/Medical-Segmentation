"""Helper utilities for preparing dataset sources without auto-downloading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Iterable


@dataclass
class DatasetSource:
    """Represent a potential dataset source selected by the user."""

    name: str
    path: Path


def build_kaggle_reference(dataset_slug: str) -> str:
    """Return a Kaggle dataset reference string for later manual download."""
    return f"kaggle datasets download -d {dataset_slug}"


def build_google_drive_reference(file_id: str) -> str:
    """Return a Google Drive sharing URL for later manual download."""
    return f"https://drive.google.com/uc?id={file_id}"


def validate_local_dataset_path(path: str | Path) -> Path:
    """Validate a user-provided local dataset path."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    return dataset_path


def collect_dataset_sources(
    kaggle_slug: str | None = None,
    google_drive_file_id: str | None = None,
    local_path: str | Path | None = None,
) -> list[DatasetSource]:
    """Collect user-provided dataset source references without downloading."""
    sources: list[DatasetSource] = []
    if kaggle_slug:
        sources.append(
            DatasetSource(
                name="kaggle",
                path=Path(build_kaggle_reference(kaggle_slug)),
            )
        )
    if google_drive_file_id:
        sources.append(
            DatasetSource(
                name="google_drive",
                path=Path(build_google_drive_reference(google_drive_file_id)),
            )
        )
    if local_path:
        sources.append(
            DatasetSource(
                name="local",
                path=validate_local_dataset_path(local_path),
            )
        )
    return sources


def copy_dataset_files(
    source_images: Path,
    source_masks: Path,
    destination_images: Path,
    destination_masks: Path,
) -> int:
    """Copy a paired dataset structure into a new location."""
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_masks.mkdir(parents=True, exist_ok=True)

    copied = 0
    image_files = sorted(
        path for path in source_images.iterdir() if path.is_file()
    )
    mask_lookup = {path.stem: path for path in source_masks.iterdir() if path.is_file()}

    for image_path in image_files:
        mask_path = mask_lookup.get(image_path.stem)
        if mask_path is None:
            continue
        copy2(image_path, destination_images / image_path.name)
        copy2(mask_path, destination_masks / mask_path.name)
        copied += 1
    return copied


def describe_sources(sources: Iterable[DatasetSource]) -> list[str]:
    """Create a readable summary for the provided sources."""
    lines = []
    for source in sources:
        lines.append(f"{source.name}: {source.path}")
    return lines

