"""Convert the Kaggle LGG Brain MRI dataset into the project layout."""

from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from configs.config import BASE_DIR, DATASET_PATH, OUTPUT_PATH


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
MASK_SUFFIXES = ("_mask",)


@dataclass(frozen=True)
class ConversionRecord:
    """Store the outcome for one converted pair or skipped file."""

    patient_folder: str
    source_image: str
    source_mask: str
    output_image: str
    output_mask: str
    status: str
    details: str


class LGGDatasetConverter:
    """Convert LGG MRI patient folders into the standard project dataset."""

    def __init__(
        self,
        source_root: str | Path,
        destination_root: str | Path | None = None,
        clear_destination: bool = True,
    ) -> None:
        """Initialize the converter with source and destination paths."""
        self.source_root = Path(source_root).expanduser().resolve()
        self.destination_root = (
            Path(destination_root).expanduser().resolve()
            if destination_root is not None
            else DATASET_PATH
        )
        self.images_dir = self.destination_root / "images"
        self.masks_dir = self.destination_root / "masks"
        self.clear_destination = clear_destination
        self.output_dir = OUTPUT_PATH
        self.report_csv = self.output_dir / "lgg_conversion_report.csv"
        self.summary_txt = self.output_dir / "lgg_conversion_summary.txt"
        self.logger = build_default_logger()

    def convert(self) -> dict[str, int | Path]:
        """Run the full conversion process and write the output reports."""
        self._validate_source()
        self._prepare_destination()

        records: list[ConversionRecord] = []
        patient_folders = self._find_patient_folders()
        patients_scanned = len(patient_folders)
        images_found = 0
        masks_found = 0
        valid_pairs = 0
        skipped_files = 0

        self.logger.info("Patients discovered: %d", patients_scanned)
        self.logger.info("Starting LGG conversion...")

        for index, patient_folder in enumerate(patient_folders, start=1):
            print(f"Scanning patient {index}/{patients_scanned}: {patient_folder.name}")
            patient_records, patient_counts = self._process_patient_folder(
                patient_folder,
                valid_pairs_start=valid_pairs,
            )
            records.extend(patient_records)
            images_found += patient_counts["images_found"]
            masks_found += patient_counts["masks_found"]
            valid_pairs += patient_counts["valid_pairs"]
            skipped_files += patient_counts["skipped_files"]

        self._write_report(records)
        self._write_summary(
            patients_scanned=patients_scanned,
            images_found=images_found,
            masks_found=masks_found,
            valid_pairs=valid_pairs,
            skipped_files=skipped_files,
        )

        result = {
            "patients_scanned": patients_scanned,
            "images_found": images_found,
            "masks_found": masks_found,
            "valid_pairs": valid_pairs,
            "skipped_files": skipped_files,
            "report_csv": self.report_csv,
            "summary_txt": self.summary_txt,
            "images_dir": self.images_dir,
            "masks_dir": self.masks_dir,
        }

        self.logger.info("LGG conversion completed successfully.")
        return result

    def _validate_source(self) -> None:
        """Ensure the LGG dataset root exists before conversion starts."""
        if not self.source_root.exists():
            raise FileNotFoundError(
                f"LGG dataset path does not exist: {self.source_root}"
            )

    def _prepare_destination(self) -> None:
        """Create or refresh the project dataset folders."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.clear_destination:
            return

        self._clear_folder(self.images_dir)
        self._clear_folder(self.masks_dir)

    def _clear_folder(self, folder: Path) -> None:
        """Remove all files from a destination folder without deleting it."""
        if not folder.exists():
            return

        for path in folder.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    def _find_patient_folders(self) -> list[Path]:
        """Locate every TCGA patient folder recursively."""
        patient_folders = {
            path.parent
            for path in self.source_root.rglob("*")
            if path.is_file() and self._is_supported_source_file(path)
        }
        return sorted(patient_folders)

    def _is_supported_source_file(self, path: Path) -> bool:
        """Return True for image files that should be considered."""
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        name = path.name.lower()
        return not name.endswith(".csv") and name != "readme.md"

    def _process_patient_folder(
        self,
        patient_folder: Path,
        valid_pairs_start: int,
    ) -> tuple[list[ConversionRecord], dict[str, int]]:
        """Convert all valid pairs found inside one patient folder."""
        records: list[ConversionRecord] = []
        images: dict[str, Path] = {}
        masks: dict[str, Path] = {}
        skipped_files = 0

        for file_path in sorted(patient_folder.iterdir()):
            if not file_path.is_file():
                continue
            if not self._is_supported_source_file(file_path):
                continue

            try:
                self._verify_image(file_path)
            except (UnidentifiedImageError, OSError, ValueError):
                skipped_files += 1
                records.append(
                    ConversionRecord(
                        patient_folder=patient_folder.name,
                        source_image=file_path.name,
                        source_mask="",
                        output_image="",
                        output_mask="",
                        status="skipped",
                        details="Corrupted or unreadable file",
                    )
                )
                continue

            sample_key = self._sample_key(file_path.stem)
            if self._is_mask_file(file_path):
                masks[sample_key] = file_path
            else:
                images[sample_key] = file_path

        paired_keys = sorted(set(images) & set(masks))
        images_found = len(images)
        masks_found = len(masks)
        valid_pairs = 0

        for local_index, sample_key in enumerate(paired_keys, start=1):
            image_path = images[sample_key]
            mask_path = masks[sample_key]
            global_index = valid_pairs_start + valid_pairs + 1
            output_image_name = f"image_{global_index:06d}.png"
            output_mask_name = f"mask_{global_index:06d}.png"
            output_image_path = self.images_dir / output_image_name
            output_mask_path = self.masks_dir / output_mask_name

            try:
                self._convert_image(image_path, output_image_path)
                self._convert_mask(mask_path, output_mask_path)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                skipped_files += 1
                records.append(
                    ConversionRecord(
                        patient_folder=patient_folder.name,
                        source_image=image_path.name,
                        source_mask=mask_path.name,
                        output_image="",
                        output_mask="",
                        status="skipped",
                        details=f"Failed to convert pair: {error}",
                    )
                )
                continue

            valid_pairs += 1
            records.append(
                ConversionRecord(
                    patient_folder=patient_folder.name,
                    source_image=image_path.name,
                    source_mask=mask_path.name,
                    output_image=output_image_name,
                    output_mask=output_mask_name,
                    status="converted",
                    details="",
                )
            )

            print(
                f"  Converted pair {local_index}/{len(paired_keys)} "
                f"-> {output_image_name}, {output_mask_name}"
            )

        unmatched_images = set(images) - set(masks)
        unmatched_masks = set(masks) - set(images)
        skipped_files += len(unmatched_images) + len(unmatched_masks)

        for sample_key in sorted(unmatched_images):
            records.append(
                ConversionRecord(
                    patient_folder=patient_folder.name,
                    source_image=images[sample_key].name,
                    source_mask="",
                    output_image="",
                    output_mask="",
                    status="skipped",
                    details="No matching mask found",
                )
            )
        for sample_key in sorted(unmatched_masks):
            records.append(
                ConversionRecord(
                    patient_folder=patient_folder.name,
                    source_image="",
                    source_mask=masks[sample_key].name,
                    output_image="",
                    output_mask="",
                    status="skipped",
                    details="No matching image found",
                )
            )

        return records, {
            "images_found": images_found,
            "masks_found": masks_found,
            "valid_pairs": valid_pairs,
            "skipped_files": skipped_files,
        }

    def _is_mask_file(self, path: Path) -> bool:
        """Detect whether a source file is a mask file."""
        stem_lower = path.stem.lower()
        return any(stem_lower.endswith(suffix) for suffix in MASK_SUFFIXES)

    def _sample_key(self, stem: str) -> str:
        """Normalize a filename stem so images and masks can be paired."""
        stem_lower = stem.lower()
        for suffix in MASK_SUFFIXES:
            if stem_lower.endswith(suffix):
                return stem_lower[: -len(suffix)]
        return stem_lower

    def _verify_image(self, path: Path) -> None:
        """Open the image once to ensure it is not corrupted."""
        with Image.open(path) as image:
            image.verify()

    def _convert_image(self, source: Path, destination: Path) -> None:
        """Convert a source image to PNG format."""
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.save(destination, format="PNG")

    def _convert_mask(self, source: Path, destination: Path) -> None:
        """Convert a source mask to grayscale PNG format."""
        with Image.open(source) as image:
            image = image.convert("L")
            image.save(destination, format="PNG")

    def _write_report(self, records: Iterable[ConversionRecord]) -> None:
        """Save the detailed per-file conversion report."""
        with self.report_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "patient_folder",
                    "source_image",
                    "source_mask",
                    "output_image",
                    "output_mask",
                    "status",
                    "details",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "patient_folder": record.patient_folder,
                        "source_image": record.source_image,
                        "source_mask": record.source_mask,
                        "output_image": record.output_image,
                        "output_mask": record.output_mask,
                        "status": record.status,
                        "details": record.details,
                    }
                )

    def _write_summary(
        self,
        patients_scanned: int,
        images_found: int,
        masks_found: int,
        valid_pairs: int,
        skipped_files: int,
    ) -> None:
        """Write a plain-text summary of the conversion results."""
        summary = (
            "LGG Dataset Conversion Summary\n"
            "==============================\n"
            f"Source Root : {self.source_root}\n"
            f"Destination Root : {self.destination_root}\n"
            f"Patients Scanned : {patients_scanned}\n"
            f"Images Found : {images_found}\n"
            f"Masks Found : {masks_found}\n"
            f"Valid Pairs : {valid_pairs}\n"
            f"Skipped Files : {skipped_files}\n"
            f"Images Output : {self.images_dir}\n"
            f"Masks Output : {self.masks_dir}\n"
        )
        self.summary_txt.write_text(summary, encoding="utf-8")


def convert_lgg_dataset(
    source_root: str | Path,
    destination_root: str | Path | None = None,
    clear_destination: bool = True,
) -> dict[str, int | Path]:
    """Convenience function for converting the LGG dataset."""
    converter = LGGDatasetConverter(
        source_root=source_root,
        destination_root=destination_root,
        clear_destination=clear_destination,
    )
    return converter.convert()


def build_default_logger() -> logging.Logger:
    """Create a logger configured for console output."""
    logger = logging.getLogger("lgg_dataset_converter")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger
