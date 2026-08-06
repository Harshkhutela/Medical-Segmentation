"""Smoke test for the large-dataset utilities and training workflow."""

from configs.dataset_config import get_dataset_configuration
from utils.dataset_splitter import DatasetSplitManager
from utils.training_helpers import CheckpointManager, HistoryLogger


def main() -> None:
    """Verify that the large-dataset helpers can discover and prepare data."""
    config = get_dataset_configuration()
    print("Large Dataset Workflow Test\n")
    print(f"Dataset Root : {config.dataset_root}")
    print(f"Dataset Name : {config.dataset_name}")

    split_manager = DatasetSplitManager(config)
    split_paths = split_manager.ensure_split_layout()
    print(f"Train Images : {split_paths.train_images}")
    print(f"Train Masks : {split_paths.train_masks}")
    print(f"Validation Images : {split_paths.val_images}")
    print(f"Validation Masks : {split_paths.val_masks}")

    history_logger = HistoryLogger(prefix="large")
    checkpoint_manager = CheckpointManager()
    print(f"History CSV : {history_logger.paths.csv_path}")
    print(f"History JSON : {history_logger.paths.json_path}")
    print("Available Checkpoints:")
    for checkpoint in checkpoint_manager.list_available_checkpoints():
        print(checkpoint)

    print("\nLarge Dataset Workflow Ready")


if __name__ == "__main__":
    main()
