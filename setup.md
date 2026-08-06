# Setup Guide

This guide explains how to run the project on a new computer from scratch.

## 1. Install prerequisites

- Python 3.10 or newer
- Git
- A code editor such as VS Code

## 2. Clone the repository

```bash
git clone <your-repository-url>
cd Medical_Segmentation
```

## 3. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Prepare the dataset

Put the dataset inside the `dataset/` folder.

Expected labeled dataset structure:

```text
dataset/
|-- images/
`-- masks/
```

If you are using the LGG Brain MRI dataset, run:

```bash
python test_lgg_converter.py
```

This prepares the dataset in the correct `dataset/images` and `dataset/masks` format.

If the train/validation split needs to be regenerated:

```bash
python rebuild_dataset_split.py
```

## 6. Verify the installation

You can run the provided test scripts:

```bash
python test_model.py
python test_dataset.py
python test_xai.py
```

If a script is not available in your copy of the project, run the matching module directly from the dashboard or command line.

## 7. Train the model

```bash
python train.py
```

For large datasets:

```bash
python train_large_dataset.py
```

## 8. Run inference

```bash
python inference.py
```

## 9. Launch the dashboard

```bash
streamlit run app.py
```

## 10. Run benchmark and XAI

Benchmark:

```bash
python -c "from benchmarking import run_benchmark; run_benchmark()"
```

or use the Benchmark page inside the dashboard.

XAI:

```bash
python test_xai.py
```

## Notes

- Do not commit `dataset/`, `checkpoints/`, `outputs/`, or `logs/` to Git.
- Make sure the checkpoint `checkpoints/best_model.pth` exists before running inference, benchmark, or XAI.
- The dashboard automatically reads generated outputs when they are available.
