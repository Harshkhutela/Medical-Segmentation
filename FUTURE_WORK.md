# Future Work

## Attention U-Net

An Attention U-Net could improve feature selection during decoding by emphasizing relevant spatial regions and suppressing irrelevant background activations. This is useful in medical images where target structures are often small or ambiguous.

## Vision Transformers

Transformer-based segmentation models can capture broader context than standard CNNs. They may improve performance on complex anatomical structures, especially when long-range dependencies matter.

## Better Uncertainty Estimation

The current active learning approach uses heuristic uncertainty scores. Future work could include Monte Carlo dropout, deep ensembles, test-time augmentation, or Bayesian methods for more reliable sample selection.

## Multi-Class Segmentation

The current project is binary segmentation oriented. Extending the pipeline to multiple classes would make it more useful for segmenting organs, lesions, and substructures simultaneously.

## Clinical Deployment

Future work could package the model into a clinical tool with robust preprocessing, secure storage, audit logs, and integration into PACS or hospital workflows.

## Additional Research Directions

- Domain adaptation across scanners and hospitals
- Better calibration of pseudo-label confidence
- Curriculum learning for difficult cases
- Self-supervised pretraining on unlabeled scans
- Interactive annotation tools for faster expert review

