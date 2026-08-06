"""Smoke test for the LGG dataset converter."""

from __future__ import annotations

import os
from pathlib import Path

from utils.lgg_dataset_converter import LGGDatasetConverter, build_default_logger


def _get_dataset_path() -> Path:
    """Read the LGG dataset root from the environment or prompt the user."""
    env_path = os.getenv("MEDSEG_LGG_DATASET")
    if env_path:
        return Path(env_path).expanduser()

    entered_path = input("LGG dataset path: ").strip().strip('"')
    if not entered_path:
        raise ValueError("No dataset path was provided.")
    return Path(entered_path).expanduser()


def main() -> None:
    """Run the LGG dataset conversion and verify the prepared dataset."""
    print("LGG Dataset Converter\n")
    source_root = _get_dataset_path()
    logger = build_default_logger()

    converter = LGGDatasetConverter(
        source_root=source_root,
        clear_destination=True,
    )
    converter.logger = logger

    result = converter.convert()

    images_dir = result["images_dir"]
    masks_dir = result["masks_dir"]
    image_count = len([path for path in Path(images_dir).iterdir() if path.is_file()])
    mask_count = len([path for path in Path(masks_dir).iterdir() if path.is_file()])

    print("\nVerification")
    print(f"dataset/images : {image_count}")
    print(f"dataset/masks : {mask_count}")

    if image_count != mask_count:
        raise RuntimeError(
            "dataset/images and dataset/masks do not contain the same number of files."
        )

    print("Verification passed.")
    print(f"Report CSV : {result['report_csv']}")
    print(f"Summary TXT : {result['summary_txt']}")


if __name__ == "__main__":
    main()
