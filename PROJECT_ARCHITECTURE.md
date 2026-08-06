# Project Architecture

## Overview

The project is organized into small, focused modules. Each folder has a clear responsibility so that the system can be maintained, extended, and reused easily.

## Root Files

- `train.py`: Trains the U-Net on labeled data and saves the best checkpoint.
- `inference.py`: Runs prediction on one sample and saves a visual result.
- `test_model.py`: Verifies the U-Net architecture with a dummy tensor.
- `test_dataset.py`: Verifies dataset loading.
- `test_active_learning.py`: Scores unlabeled images by uncertainty.
- `test_semi_supervised.py`: Generates pseudo labels and retraining data.
- `test_evaluation.py`: Reads logs, exports evaluation CSV, and generates plots.
- `pipeline.py`: Runs the complete end-to-end research workflow.

## `configs/`

- `config.py`: Central place for project paths, image size, batch size, learning rate, epochs, and device selection.

## `dataset/`

- `images/`: Labeled input images.
- `masks/`: Ground truth masks for labeled images.
- `unlabeled/`: Images without annotations.
- `pseudo_masks/`: Automatically generated pseudo labels.
- `retraining/`: Combined dataset used for retraining.

This folder is the data backbone of the project.

## `models/`

- `__init__.py`: Exposes the model classes for import.
- `blocks.py`: Reusable U-Net building blocks such as DoubleConv, DownBlock, and UpBlock.
- `unet.py`: Full U-Net encoder-decoder architecture.

This folder isolates the network definition from all training and inference logic.

## `utils/`

- `__init__.py`: Package initializer.
- `dataset.py`: Dataset loader that matches image-mask pairs by filename stem.
- `losses.py`: Dice loss and combined BCE + Dice loss.
- `metrics.py`: Dice score, IoU score, and pixel accuracy.
- `visualize.py`: Matplotlib-based visualization utilities.

This folder contains reusable helpers shared across the pipeline.

## `active_learning/`

- `__init__.py`: Package initializer.
- `uncertainty.py`: Implements least confidence, entropy, and margin sampling.
- `sampler.py`: Loads the model and computes uncertainty scores for unlabeled images.
- `query_strategy.py`: Selects the top-K most uncertain images.

This folder supports annotation planning.

## `semi_supervised/`

- `__init__.py`: Package initializer.
- `pseudo_label.py`: Generates pseudo masks for high-confidence unlabeled images.
- `retrain.py`: Builds a merged retraining dataset from labeled and pseudo-labeled data.

This folder supports data expansion without manual annotation.

## `evaluation/`

- `__init__.py`: Package initializer.
- `evaluator.py`: Reads experiment logs, computes summary statistics, and exports CSV.
- `plots.py`: Generates publication-style metric plots.

This folder supports experiment reporting and analysis.

## `checkpoints/`

Stores trained model weights, especially `best_model.pth`.

## `outputs/`

Stores generated artifacts such as:

- `history.csv`
- `history.json`
- `experiment_results.csv`
- `active_learning_scores.csv`
- `semi_supervised_report.txt`
- `predictions/prediction_result.png`
- `plots/`

## `logs/`

Stores orchestration logs such as `logs/pipeline_log.txt`.

## Execution Flow

1. `train.py` creates a checkpoint and logs metrics.
2. `inference.py` uses the checkpoint for prediction.
3. `active_learning/` scores unlabeled data.
4. `semi_supervised/` creates pseudo labels and retraining data.
5. `evaluation/` summarizes the results and creates plots.
6. `pipeline.py` chains everything into one reproducible workflow.

