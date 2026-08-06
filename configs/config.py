import torch
from pathlib import Path


# ==========================
# PROJECT PATHS
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset"

IMAGE_FOLDER = DATASET_PATH / "images"

MASK_FOLDER = DATASET_PATH / "masks"

CHECKPOINT_PATH = BASE_DIR / "checkpoints"

OUTPUT_PATH = BASE_DIR / "outputs"


# ==========================
# IMAGE SETTINGS
# ==========================

IMAGE_SIZE = 256

CHANNELS = 3


# ==========================
# TRAINING SETTINGS
# ==========================

BATCH_SIZE = 4

LEARNING_RATE = 0.001

EPOCHS = 30


# ==========================
# DEVICE
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")