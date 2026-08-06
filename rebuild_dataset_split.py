"""Standalone helper to rebuild the train/validation dataset split."""

from __future__ import annotations

from configs.dataset_config import get_dataset_configuration
from utils.dataset_splitter import DatasetSplitManager


def main() -> int:
    """Rebuild the split folders from the source image and mask directories."""
    try:
        manager = DatasetSplitManager(get_dataset_configuration())
        manager.ensure_split_layout()
    except Exception as error:  # pragma: no cover - friendly CLI behavior
        print(f"Failed to rebuild dataset split: {error}")
        return 1

    print("Dataset split rebuilt successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
