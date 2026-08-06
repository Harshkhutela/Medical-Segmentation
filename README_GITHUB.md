# Medical_Segmentation

## Project Title

Medical_Segmentation is a PyTorch-based medical image segmentation project for binary segmentation of MRI scans using U-Net, active learning, semi-supervised learning, benchmarking, evaluation, and explainable AI.

## Features

- U-Net segmentation model for binary mask prediction
- Training pipeline with checkpointing and history logging
- Inference pipeline for uploaded or dataset images
- Active learning for uncertainty-based sample selection
- Semi-supervised pseudo-label generation and retraining preparation
- Benchmarking against simple baselines
- Explainable AI visualizations with Grad-CAM-style heatmaps
- Streamlit dashboard for interactive research demonstrations

## Folder Structure

```text
Medical_Segmentation/
|-- app.py
|-- configs/
|-- dataset/
|-- models/
|-- utils/
|-- active_learning/
|-- semi_supervised/
|-- benchmarking/
|-- evaluation/
|-- xai/
|-- dashboard/
|-- outputs/
|-- checkpoints/
`-- logs/
```

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

## Download Dataset

Place the dataset in the `dataset/` folder using the expected structure:

```text
dataset/
├── images/
├── masks/
├── unlabeled/
└── pseudo_masks/
```

For the LGG Brain MRI dataset, use the provided conversion utility before training.

## Convert Dataset

If you are using the LGG Brain MRI dataset, run the dataset conversion helper to prepare `dataset/images` and `dataset/masks`.

```bash
python test_lgg_converter.py
```

## Train Model

Run the standard training pipeline:

```bash
python train.py
```

For large datasets:

```bash
python train_large_dataset.py
```

## Launch Dashboard

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

## Run Inference

Run standalone inference on a sample image:

```bash
python inference.py
```

## Run XAI

Generate Grad-CAM-style explanations and attention visualizations:

```bash
python test_xai.py
```

## Run Benchmark

Compare the trained U-Net against simple baselines:

```bash
python -c "from benchmarking import run_benchmark; run_benchmark()"
```

You can also use the Benchmark page in the Streamlit dashboard.

## Future Work

- Multi-class segmentation
- Attention U-Net and Transformer-based segmentation models
- Better uncertainty estimation for active learning
- Stronger explanation methods for clinical review
- Deployment-oriented inference and reporting tools
