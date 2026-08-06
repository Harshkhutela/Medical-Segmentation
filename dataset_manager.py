"""Dataset management utilities for medical segmentation datasets."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs.config import BASE_DIR, OUTPUT_PATH
from utils.dataset_statistics import (
    SUPPORTED_EXTENSIONS,
    average_dimensions,
    get_foreground_background_percentages,
    is_supported_image,
    read_image_shape,
    read_mask_array,
    summarize_resolutions,
)


@dataclass
class DatasetIssue:
    """Represent a dataset validation issue."""

    issue_type: str
    name: str
    details: str


class DatasetManager:
    """Scan, validate, report, and visualize an ``images/masks`` dataset."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize the manager with the dataset and output paths."""
        self.dataset_root = Path(dataset_root) if dataset_root else BASE_DIR / "dataset"
        self.images_dir = self.dataset_root / "images"
        self.masks_dir = self.dataset_root / "masks"
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_PATH
        self.report_csv = self.output_dir / "dataset_report.csv"
        self.summary_txt = self.output_dir / "dataset_summary.txt"
        self.preview_png = self.output_dir / "dataset_preview.png"

    def analyze(self) -> dict[str, Any]:
        """Run the full dataset analysis workflow."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        image_files, image_issues = self._collect_files_with_issues(
            self.images_dir,
            "image",
        )
        mask_files, mask_issues = self._collect_files_with_issues(
            self.masks_dir,
            "mask",
        )

        valid_pairs, pair_issues = self._match_pairs(image_files, mask_files)
        issues = image_issues + mask_issues + pair_issues
        statistics = self._compute_statistics(valid_pairs)

        self._write_csv_report(valid_pairs, issues, statistics)
        self._write_summary(valid_pairs, issues, statistics)
        self._write_preview(valid_pairs)

        return {
            "total_images": len(image_files),
            "total_masks": len(mask_files),
            "total_valid_pairs": len(valid_pairs),
            "issues": issues,
            "statistics": statistics,
            "report_csv": self.report_csv,
            "summary_txt": self.summary_txt,
            "preview_png": self.preview_png,
        }

    def _collect_files_with_issues(
        self,
        folder: Path,
        kind: str,
    ) -> tuple[list[Path], list[DatasetIssue]]:
        """Return supported files and report unsupported ones as issues."""
        if not folder.is_dir():
            return [], []

        supported_files: list[Path] = []
        issues: list[DatasetIssue] = []
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if is_supported_image(path):
                supported_files.append(path)
                continue
            issues.append(
                DatasetIssue(
                    issue_type="unsupported_format",
                    name=path.name,
                    details=f"Unsupported {kind} format: {path.suffix.lower()}",
                )
            )

        return supported_files, issues

    def _match_pairs(
        self,
        image_files: list[Path],
        mask_files: list[Path],
    ) -> tuple[list[dict[str, Any]], list[DatasetIssue]]:
        """Validate pairings and collect issues such as mismatches and duplicates."""
        issues: list[DatasetIssue] = []
        image_groups = self._group_by_stem(image_files, "image")
        mask_groups = self._group_by_stem(mask_files, "mask")

        valid_pairs: list[dict[str, Any]] = []

        all_stems = sorted(set(image_groups) | set(mask_groups))
        for stem in all_stems:
            image_group = image_groups.get(stem, [])
            mask_group = mask_groups.get(stem, [])

            if len(image_group) > 1:
                issues.append(
                    DatasetIssue(
                        issue_type="duplicate_image",
                        name=stem,
                        details=", ".join(path.name for path in image_group),
                    )
                )
            if len(mask_group) > 1:
                issues.append(
                    DatasetIssue(
                        issue_type="duplicate_mask",
                        name=stem,
                        details=", ".join(path.name for path in mask_group),
                    )
                )

            image_path = image_group[0] if image_group else None
            mask_path = mask_group[0] if mask_group else None
            if image_path is None:
                issues.append(
                    DatasetIssue(
                        issue_type="missing_image",
                        name=stem,
                        details=f"No image found for mask {mask_path.name}" if mask_path else "Missing image",
                    )
                )
                continue
            if mask_path is None:
                issues.append(
                    DatasetIssue(
                        issue_type="missing_mask",
                        name=stem,
                        details=f"No mask found for image {image_path.name}",
                    )
                )
                continue

            image_shape = read_image_shape(image_path)
            mask_array = read_mask_array(mask_path)
            if image_shape is None:
                issues.append(
                    DatasetIssue(
                        issue_type="unreadable_image",
                        name=image_path.name,
                        details="Unable to read image file",
                    )
                )
                continue
            if mask_array is None:
                issues.append(
                    DatasetIssue(
                        issue_type="unreadable_mask",
                        name=mask_path.name,
                        details="Unable to read mask file",
                    )
                )
                continue

            image_width, image_height = image_shape
            mask_height, mask_width = mask_array.shape[:2]
            if (image_width, image_height) != (mask_width, mask_height):
                issues.append(
                    DatasetIssue(
                        issue_type="image_mask_mismatch",
                        name=stem,
                        details=(
                            f"Image {image_width}x{image_height} does not match "
                            f"mask {mask_width}x{mask_height}"
                        ),
                    )
                )

            foreground_percentage, background_percentage = (
                get_foreground_background_percentages(mask_array)
            )

            valid_pairs.append(
                {
                    "stem": stem,
                    "image_name": image_path.name,
                    "mask_name": mask_path.name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "mask_width": mask_width,
                    "mask_height": mask_height,
                    "foreground_percentage": foreground_percentage,
                    "background_percentage": background_percentage,
                }
            )

        return valid_pairs, issues

    def _group_by_stem(
        self,
        paths: list[Path],
        kind: str,
    ) -> dict[str, list[Path]]:
        """Group image or mask paths by filename stem."""
        grouped: dict[str, list[Path]] = {}
        for path in paths:
            grouped.setdefault(path.stem, []).append(path)

        for stem, items in grouped.items():
            if any(path.suffix.lower() not in SUPPORTED_EXTENSIONS for path in items):
                continue
            grouped[stem] = sorted(items)
        return grouped

    def _compute_statistics(self, valid_pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute dataset-level statistics from valid pairs."""
        image_resolutions = [
            (item["image_width"], item["image_height"]) for item in valid_pairs
        ]
        mask_resolutions = [
            (item["mask_width"], item["mask_height"]) for item in valid_pairs
        ]

        image_summary = summarize_resolutions(image_resolutions)
        mask_summary = summarize_resolutions(mask_resolutions)
        average_width, average_height = average_dimensions(image_resolutions)

        total_foreground = 0
        total_background = 0
        for item in valid_pairs:
            mask = read_mask_array(item["mask_path"])
            if mask is None:
                continue
            foreground = int(np.count_nonzero(mask))
            background = int(mask.size - foreground)
            total_foreground += foreground
            total_background += background

        total_pixels = max(total_foreground + total_background, 1)
        foreground_percentage = (total_foreground / total_pixels) * 100.0
        background_percentage = (total_background / total_pixels) * 100.0

        return {
            "image_resolution_distribution": image_summary.counts,
            "mask_resolution_distribution": mask_summary.counts,
            "average_width": average_width,
            "average_height": average_height,
            "foreground_percentage": foreground_percentage,
            "background_percentage": background_percentage,
        }

    def _write_csv_report(
        self,
        valid_pairs: list[dict[str, Any]],
        issues: list[DatasetIssue],
        statistics: dict[str, Any],
    ) -> None:
        """Save a detailed per-sample CSV report."""
        fieldnames = [
            "record_type",
            "name",
            "image_name",
            "mask_name",
            "image_width",
            "image_height",
            "mask_width",
            "mask_height",
            "foreground_percentage",
            "background_percentage",
            "details",
        ]

        with self.report_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for pair in valid_pairs:
                writer.writerow(
                    {
                        "record_type": "pair",
                        "name": pair["stem"],
                        "image_name": pair["image_name"],
                        "mask_name": pair["mask_name"],
                        "image_width": pair["image_width"],
                        "image_height": pair["image_height"],
                        "mask_width": pair["mask_width"],
                        "mask_height": pair["mask_height"],
                        "foreground_percentage": f"{pair['foreground_percentage']:.4f}",
                        "background_percentage": f"{pair['background_percentage']:.4f}",
                        "details": "",
                    }
                )

            for issue in issues:
                writer.writerow(
                    {
                        "record_type": "issue",
                        "name": issue.name,
                        "image_name": "",
                        "mask_name": "",
                        "image_width": "",
                        "image_height": "",
                        "mask_width": "",
                        "mask_height": "",
                        "foreground_percentage": "",
                        "background_percentage": "",
                        "details": f"{issue.issue_type}: {issue.details}",
                    }
                )

            writer.writerow(
                {
                    "record_type": "summary",
                    "name": "dataset_summary",
                    "image_name": "",
                    "mask_name": "",
                    "image_width": statistics["average_width"],
                    "image_height": statistics["average_height"],
                    "mask_width": "",
                    "mask_height": "",
                    "foreground_percentage": f"{statistics['foreground_percentage']:.4f}",
                    "background_percentage": f"{statistics['background_percentage']:.4f}",
                    "details": "Aggregated dataset summary",
                }
            )

    def _write_summary(
        self,
        valid_pairs: list[dict[str, Any]],
        issues: list[DatasetIssue],
        statistics: dict[str, Any],
    ) -> None:
        """Save a human-readable dataset summary report."""
        lines = [
            "Dataset Summary",
            "================",
            f"Total Images : {len(self._collect_files_with_issues(self.images_dir, 'image')[0])}",
            f"Total Masks : {len(self._collect_files_with_issues(self.masks_dir, 'mask')[0])}",
            f"Total Valid Pairs : {len(valid_pairs)}",
            f"Average Width : {statistics['average_width']:.2f}",
            f"Average Height : {statistics['average_height']:.2f}",
            f"Foreground Percentage : {statistics['foreground_percentage']:.2f}%",
            f"Background Percentage : {statistics['background_percentage']:.2f}%",
            "",
            "Image Resolution Distribution",
        ]

        image_distribution = statistics["image_resolution_distribution"]
        if image_distribution:
            for resolution, count in sorted(image_distribution.items()):
                lines.append(f"  {resolution} -> {count}")
        else:
            lines.append("  None")

        lines.append("")
        lines.append("Mask Resolution Distribution")
        mask_distribution = statistics["mask_resolution_distribution"]
        if mask_distribution:
            for resolution, count in sorted(mask_distribution.items()):
                lines.append(f"  {resolution} -> {count}")
        else:
            lines.append("  None")

        lines.append("")
        lines.append("Issues")
        if issues:
            for issue in issues:
                lines.append(f"  {issue.issue_type} | {issue.name} | {issue.details}")
        else:
            lines.append("  None")

        self.summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_preview(self, valid_pairs: list[dict[str, Any]]) -> None:
        """Generate a 9-sample visual preview when enough samples exist."""
        if not valid_pairs:
            return

        sample_count = min(9, len(valid_pairs))
        selected_pairs = random.sample(valid_pairs, k=sample_count)

        figure, axes = plt.subplots(
            sample_count,
            3,
            figsize=(12, 4 * sample_count),
            dpi=200,
        )
        if sample_count == 1:
            axes = np.expand_dims(axes, axis=0)

        column_titles = ["Original", "Mask", "Overlay"]

        for row_index, pair in enumerate(selected_pairs):
            image = cv2.imread(str(pair["image_path"]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = image.copy()
            overlay_mask = np.zeros_like(image)
            overlay_mask[:, :, 0] = 255
            mask_binary = mask > 0
            overlay[mask_binary] = (
                0.7 * overlay[mask_binary] + 0.3 * overlay_mask[mask_binary]
            ).astype(np.uint8)

            axes[row_index, 0].imshow(image)
            axes[row_index, 1].imshow(mask, cmap="gray")
            axes[row_index, 2].imshow(overlay)

            for col_index in range(3):
                axes[row_index, col_index].axis("off")
                if row_index == 0:
                    axes[row_index, col_index].set_title(column_titles[col_index])

            axes[row_index, 0].set_ylabel(pair["stem"], rotation=0, labelpad=50)

        figure.tight_layout()
        figure.savefig(self.preview_png, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    """Allow the dataset manager to be run as a standalone script."""
    manager = DatasetManager()
    result = manager.analyze()
    print("Dataset Manager Completed")
    print(f"Total Images : {result['total_images']}")
    print(f"Total Masks : {result['total_masks']}")
    print(f"Total Valid Pairs : {result['total_valid_pairs']}")
    print(f"CSV Report : {result['report_csv']}")
    print(f"Summary Report : {result['summary_txt']}")
    print(f"Preview : {result['preview_png']}")


if __name__ == "__main__":
    main()
