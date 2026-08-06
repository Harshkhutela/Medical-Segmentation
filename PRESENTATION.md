# Presentation Outline

## Slide 1: Title

- Medical Image Segmentation with U-Net, Active Learning, and Semi-Supervised Learning
- Name, roll number, department, and guide details

## Slide 2: Problem Statement

- Manual annotation is expensive
- Medical segmentation requires pixel-level labels
- Need efficient learning with limited annotations

## Slide 3: Project Objectives

- Build a U-Net segmentation model
- Train and evaluate on medical images
- Use uncertainty for sample selection
- Use pseudo-labeling to expand data

## Slide 4: Dataset

- Labeled images and masks
- Unlabeled images
- Folder structure
- Preprocessing and pairing strategy

## Slide 5: Model Architecture

- U-Net encoder-decoder design
- DoubleConv, DownBlock, UpBlock
- Skip connections
- Binary output with raw logits

## Slide 6: Training Pipeline

- Dataset split
- Loss function
- Optimizer
- Checkpoint saving
- History logging

## Slide 7: Inference and Visualization

- Load checkpoint
- Predict one sample
- Apply sigmoid and threshold
- Show original image, ground truth, and prediction

## Slide 8: Active Learning

- Uncertainty-based scoring
- Least confidence
- Entropy
- Margin sampling
- Top-K selection

## Slide 9: Semi-Supervised Learning

- Pseudo-label generation
- Confidence threshold
- Pseudo mask storage
- Retraining dataset creation

## Slide 10: Evaluation

- Dice and IoU curves
- CSV export
- Summary statistics
- Experiment reporting

## Slide 11: Results and Discussion

- Best checkpoint
- Qualitative visualization
- Benefits of active and semi-supervised learning
- Observed limitations

## Slide 12: Conclusion and Future Work

- End-to-end modular research pipeline
- Better architectures
- Better uncertainty estimation
- Multi-class and deployment directions

