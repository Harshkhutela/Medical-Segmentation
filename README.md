# Medical Segmentation

## Project Overview

This project is a complete PyTorch-based medical image segmentation workflow built around a U-Net architecture. It includes dataset loading, training, inference, uncertainty-based active learning, semi-supervised pseudo-labeling, evaluation, experiment logging, and a unified research pipeline.

The system is designed for binary segmentation of medical images, where the goal is to predict a foreground mask for each input image.

## Problem Statement

Manual annotation of medical images is expensive, slow, and requires clinical expertise. At the same time, segmentation models need large, high-quality labeled datasets to perform well. This project addresses that gap by combining supervised learning with active learning and semi-supervised learning so that the model can improve with fewer expert annotations.

## Objectives

- Build a clean U-Net segmentation model in PyTorch.
- Train the model on paired image-mask data.
- Evaluate predictions using Dice and IoU.
- Rank unlabeled images by uncertainty.
- Generate pseudo labels for confident unlabeled images.
- Prepare a retraining dataset using labeled and pseudo-labeled samples.
- Compare experiments and generate plots.
- Run the entire workflow through one orchestration pipeline.

## Folder Structure

```text
Medical_Segmentation/
|-- configs/
|   `-- config.py
|-- dataset/
|   |-- images/
|   |-- masks/
|   |-- unlabeled/
|   |-- pseudo_masks/
|   `-- retraining/
|-- models/
|   |-- __init__.py
|   |-- blocks.py
|   `-- unet.py
|-- utils/
|   |-- __init__.py
|   |-- dataset.py
|   |-- losses.py
|   |-- metrics.py
|   |-- visualize.py
|-- active_learning/
|   |-- __init__.py
|   |-- uncertainty.py
|   |-- sampler.py
|   `-- query_strategy.py
|-- semi_supervised/
|   |-- __init__.py
|   |-- pseudo_label.py
|   `-- retrain.py
|-- evaluation/
|   |-- __init__.py
|   |-- evaluator.py
|   `-- plots.py
|-- outputs/
|-- checkpoints/
|-- logs/
|-- train.py
|-- inference.py
|-- test_model.py
|-- test_dataset.py
|-- test_active_learning.py
|-- test_semi_supervised.py
|-- test_evaluation.py
`-- pipeline.py
```

## Installation

1. Create and activate the virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the scripts from the project root.

```bash
python train.py
python inference.py
python test_active_learning.py
python test_semi_supervised.py
python test_evaluation.py
python pipeline.py
```

## Training

The training script:

- Loads paired images and masks using `MedicalSegmentationDataset`.
- Splits the dataset into 80 percent training and 20 percent validation sets.
- Trains a U-Net using `BCEDiceLoss`.
- Saves the best checkpoint to `checkpoints/best_model.pth`.
- Logs per-epoch metrics to `outputs/history.csv` and `outputs/history.json`.

## Inference

The inference script:

- Loads the trained checkpoint.
- Selects one sample from the dataset.
- Runs the model in evaluation mode.
- Applies sigmoid and thresholding.
- Saves a visualization with original image, ground truth, and prediction.

## Active Learning

The active learning module:

- Scans `dataset/unlabeled/`.
- Scores each image using least confidence, entropy, and margin sampling.
- Sorts images from most uncertain to least uncertain.
- Saves uncertainty scores to `outputs/active_learning_scores.csv`.

## Semi-Supervised Learning

The semi-supervised module:

- Scans unlabeled images.
- Generates pseudo masks for confident predictions.
- Saves pseudo labels in `dataset/pseudo_masks/`.
- Merges labeled and pseudo-labeled pairs into `dataset/retraining/`.

## Evaluation

The evaluation module:

- Reads `outputs/history.csv` and `outputs/history.json`.
- Reads active learning and semi-supervised reports if available.
- Exports a normalized experiment CSV to `outputs/experiment_results.csv`.
- Generates training curves in `outputs/plots/`.

## Results

The project currently produces:

- A trained checkpoint in `checkpoints/best_model.pth`
- Prediction visualization in `outputs/predictions/prediction_result.png`
- Active learning scores in `outputs/active_learning_scores.csv`
- Semi-supervised summary in `outputs/semi_supervised_report.txt`
- Evaluation plots in `outputs/plots/`
- Training logs in `outputs/history.csv` and `outputs/history.json`

## Future Improvements

- Add attention-based U-Net variants.
- Explore transformer-based segmentation architectures.
- Improve uncertainty estimation strategies.
- Extend the system to multi-class segmentation.
- Add clinical validation and deployment support.

