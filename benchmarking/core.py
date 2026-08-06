"""Core benchmarking utilities for research comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.config import CHECKPOINT_PATH, DEVICE, OUTPUT_PATH
from configs.dataset_config import get_dataset_configuration
from inference import load_model
from utils.dataset import MedicalSegmentationDataset
from utils.dataset_splitter import DatasetSplitManager
from utils.losses import prepare_targets
from benchmarking.plots import save_benchmark_artifacts


BENCHMARK_RESULTS_PATH = OUTPUT_PATH / "benchmark_results.csv"
BENCHMARK_PLOTS_DIR = OUTPUT_PATH / "benchmark_plots"
BENCHMARK_TABLE_PATH = BENCHMARK_PLOTS_DIR / "benchmark_table.png"


@dataclass(frozen=True)
class BenchmarkResult:
    """Store benchmark metrics for one model or baseline."""

    model: str
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    samples: int


def _build_validation_dataset() -> MedicalSegmentationDataset:
    """Load the validation split, creating it when needed."""
    split_manager = DatasetSplitManager(get_dataset_configuration())
    split_paths = split_manager.ensure_split_layout()
    return MedicalSegmentationDataset(
        image_folder=split_paths.val_images,
        mask_folder=split_paths.val_masks,
    )


def _load_unet_model():
    """Load the trained U-Net checkpoint for benchmarking."""
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Benchmark checkpoint not found at {checkpoint_file}."
        )
    return load_model(checkpoint_file)


def _reduce_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute segmentation metrics for a batch of logits and masks."""
    predicted_masks = (torch.sigmoid(predictions) >= 0.5).float()
    target_masks = prepare_targets(targets).to(dtype=predicted_masks.dtype)

    predicted_masks = predicted_masks.flatten(start_dim=1)
    target_masks = target_masks.flatten(start_dim=1)

    intersection = (predicted_masks * target_masks).sum(dim=1)
    prediction_sum = predicted_masks.sum(dim=1)
    target_sum = target_masks.sum(dim=1)
    fp = (predicted_masks * (1 - target_masks)).sum(dim=1)
    fn = ((1 - predicted_masks) * target_masks).sum(dim=1)

    dice = (2 * intersection + 1e-8) / (prediction_sum + target_sum + 1e-8)
    iou = (intersection + 1e-8) / (prediction_sum + target_sum - intersection + 1e-8)
    precision = (intersection + 1e-8) / (intersection + fp + 1e-8)
    recall = (intersection + 1e-8) / (intersection + fn + 1e-8)
    f1 = (2 * precision * recall + 1e-8) / (precision + recall + 1e-8)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }


def _predict_unet(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Generate logits using the trained U-Net."""
    with torch.no_grad():
        return model(images.to(DEVICE))


def _predict_dummy(images: torch.Tensor) -> torch.Tensor:
    """Predict an all-background mask."""
    return torch.full(
        (images.size(0), 1, images.size(2), images.size(3)),
        -10.0,
        device=DEVICE,
    )


def _predict_random(images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Predict random logits for a non-learning baseline."""
    return torch.randn(
        (images.size(0), 1, images.size(2), images.size(3)),
        device=DEVICE,
        generator=generator,
    )


def _accumulate(
    running: dict[str, float],
    batch_scores: dict[str, float],
    batch_size: int,
) -> None:
    """Accumulate weighted batch metrics."""
    for key, value in batch_scores.items():
        running[key] = running.get(key, 0.0) + (value * batch_size)


def _finalize(running: dict[str, float], total_samples: int) -> dict[str, float]:
    """Convert accumulated sums into averages."""
    if total_samples == 0:
        return {key: 0.0 for key in running}
    return {key: value / total_samples for key, value in running.items()}


def run_benchmark(
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> pd.DataFrame:
    """Run the benchmark and save results to disk."""
    dataset = _build_validation_dataset()
    if len(dataset) == 0:
        raise ValueError("No validation samples were found for benchmarking.")

    config = get_dataset_configuration()
    loader = DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else config.num_workers,
        pin_memory=config.pin_memory and DEVICE.type == "cuda",
    )

    model = _load_unet_model()
    model.eval()

    generator = torch.Generator(device=DEVICE if DEVICE.type == "cuda" else "cpu")
    generator.manual_seed(config.seed)

    totals: dict[str, dict[str, float]] = {
        "U-Net": {},
        "Dummy Baseline": {},
        "Random Predictor": {},
    }
    sample_count = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        batch_size_value = int(images.size(0))
        sample_count += batch_size_value

        unet_logits = _predict_unet(model, images)
        dummy_logits = _predict_dummy(images)
        random_logits = _predict_random(images, generator=generator)

        _accumulate(totals["U-Net"], _reduce_batch_metrics(unet_logits, masks), batch_size_value)
        _accumulate(
            totals["Dummy Baseline"],
            _reduce_batch_metrics(dummy_logits, masks),
            batch_size_value,
        )
        _accumulate(
            totals["Random Predictor"],
            _reduce_batch_metrics(random_logits, masks),
            batch_size_value,
        )

    rows: list[BenchmarkResult] = []
    for model_name, metrics in totals.items():
        averaged = _finalize(metrics, sample_count)
        rows.append(
            BenchmarkResult(
                model=model_name,
                dice=averaged.get("dice", 0.0),
                iou=averaged.get("iou", 0.0),
                precision=averaged.get("precision", 0.0),
                recall=averaged.get("recall", 0.0),
                f1=averaged.get("f1", 0.0),
                samples=sample_count,
            )
        )

    frame = pd.DataFrame([asdict(row) for row in rows])
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(BENCHMARK_RESULTS_PATH, index=False)
    save_benchmark_artifacts(frame, BENCHMARK_PLOTS_DIR)
    return frame


def load_benchmark_results() -> pd.DataFrame:
    """Load previously saved benchmark results if available."""
    if not BENCHMARK_RESULTS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(BENCHMARK_RESULTS_PATH)
    except Exception:
        return pd.DataFrame()
