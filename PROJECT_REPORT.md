# Project Report

## Abstract

This project presents a complete medical image segmentation framework implemented in PyTorch. The system combines supervised U-Net training with active learning, semi-supervised pseudo-labeling, experiment evaluation, and a unified research pipeline. The framework is designed to reduce annotation cost while maintaining strong segmentation performance on binary medical image masks.

## Introduction

Medical image segmentation is a core task in computer-aided diagnosis, treatment planning, and biomedical image analysis. It aims to separate anatomical structures or pathological regions from the background. However, expert annotation is expensive and limited. This project addresses that challenge by integrating learning strategies that make better use of both labeled and unlabeled data.

## Literature Motivation

Segmentation models often require dense pixel-level annotations, which are costly in medical imaging. U-Net became a standard because of its encoder-decoder design and skip connections. Later work showed that active learning can reduce annotation burden by selecting informative samples, and semi-supervised learning can expand training data with pseudo labels. This project combines these ideas into one practical pipeline.

## Methodology

The workflow is organized into the following stages:

1. Load paired medical image and mask data.
2. Train a U-Net on labeled samples.
3. Run inference on unlabeled images.
4. Estimate uncertainty for each unlabeled prediction.
5. Select the most informative samples for annotation.
6. Generate pseudo labels for highly confident predictions.
7. Prepare a merged retraining dataset.
8. Retrain the model on the combined data.
9. Evaluate and visualize experiment results.

## System Architecture

The system is modular and file-based.

- `configs/config.py` stores all paths and hyperparameters.
- `utils/dataset.py` loads images and masks.
- `models/` defines the U-Net architecture.
- `train.py` handles supervised training.
- `inference.py` performs prediction and visualization.
- `active_learning/` computes uncertainty scores and selects samples.
- `semi_supervised/` generates pseudo labels and retraining data.
- `evaluation/` summarizes experiments and produces plots.
- `pipeline.py` orchestrates the full workflow.

## Dataset

The dataset is organized into:

- `dataset/images/` for labeled medical images
- `dataset/masks/` for ground truth masks
- `dataset/unlabeled/` for unlabeled images
- `dataset/pseudo_masks/` for generated pseudo labels
- `dataset/retraining/` for the combined retraining dataset

The loader pairs images and masks using matching filename stems.

## Model

The segmentation network is a U-Net with:

- An encoder path for feature extraction
- A decoder path for upsampling
- Skip connections to recover spatial detail
- Raw logits at the output layer

The model uses a binary output channel for foreground-background segmentation.

## Training

Training uses:

- `BCEDiceLoss`
- Adam optimizer
- `BATCH_SIZE`, `LEARNING_RATE`, and `EPOCHS` from config
- 80/20 train-validation split
- checkpointing of the best Dice score
- history logging to CSV and JSON

The training objective balances class imbalance handling from Dice with the stability of BCE.

## Inference

Inference loads the saved checkpoint and runs one sample through the model. The output logits are converted to probabilities using sigmoid and then thresholded at 0.5 to obtain a binary mask. A visualization is saved for qualitative inspection.

## Active Learning

Active learning scores each unlabeled image using:

- Least confidence
- Entropy
- Margin sampling

The scores are combined into a final uncertainty score. The most uncertain images are selected for annotation planning.

## Semi-Supervised Learning

Semi-supervised learning generates pseudo masks for unlabeled images when confidence exceeds a threshold. High-confidence predictions are kept and saved to `dataset/pseudo_masks/`. These pseudo-labeled samples are then merged with the labeled dataset for retraining.

## Results

The project generates:

- Training and validation curves
- Prediction visualizations
- Active learning uncertainty rankings
- Semi-supervised summaries
- Unified evaluation reports

The actual numerical results depend on the dataset and checkpoint used during experimentation.

## Limitations

- Binary segmentation only.
- Performance depends on dataset quality and size.
- Pseudo labels can introduce noise if confidence is poorly calibrated.
- Active learning ranking is based on heuristic uncertainty measures.
- The pipeline assumes consistent folder structure and file naming.

## Future Scope

- Attention U-Net or transformer-based segmentation.
- More advanced uncertainty estimation.
- Multi-class segmentation support.
- Test-time augmentation and model ensembling.
- Clinical deployment and external validation.

## Conclusion

This project provides an end-to-end medical image segmentation research framework. It is practical, modular, and extensible. By combining supervised learning, uncertainty-based sample selection, pseudo-labeling, and experiment evaluation, the system reduces manual annotation effort while supporting iterative model improvement.

