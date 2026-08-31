<div align="center">
  <h1>🧠 Medical AI Framework</h1>
  <h3>End-to-End Semi-Supervised Active Learning for Image Segmentation</h3>
  
  <p>
    A production-ready research pipeline built for analyzing medical imagery (MRI/CT). This project combines deep learning segmentation, uncertainty estimation, and explainable AI into a single, cohesive ecosystem designed to accelerate clinical research and reduce annotation costs.
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg" alt="Streamlit" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

---

## 🌟 Key Features

This framework is not just a model—it is a complete research pipeline designed to bridge the gap between raw medical data and interpretable clinical insights.

- **🔬 High-Precision Segmentation (U-Net)**: A robust encoder-decoder architecture optimized for binary segmentation of complex anatomical structures and pathologies.
- **🎯 Active Learning Engine**: Intelligently identifies the most confusing or uncertain scans in large unlabelled datasets, ensuring doctors only spend time annotating the images that actually improve the model.
- **🔄 Semi-Supervised Pseudo-Labeling**: Automatically generates high-confidence labels for easy cases, exponentially expanding the dataset without manual human effort.
- **📊 Comprehensive Benchmarking**: Automatically pits the trained neural network against dummy baselines (All-Zeros, Random) to statistically validate that the model is learning genuine physiological features.
- **👁️ Explainable AI (XAI)**: Peeks inside the "black box" using Grad-CAM heatmaps, forcing the AI to highlight exactly which pixels influenced its medical diagnosis, building clinical trust.
- **🖥️ Interactive Research Dashboard**: A beautiful, real-time Streamlit web interface for executing inference, tweaking confidence thresholds, and analyzing multi-dimensional metrics on the fly.

## 🏗️ Architecture & Structure

The repository is highly modular, following best practices for scalable machine learning engineering:

```text
Medical_Segmentation/
├── active_learning/    # Uncertainty estimation (Entropy, Margin, Least Confidence)
├── benchmarking/       # Automated baseline testing & metric validation
├── dashboard/          # Interactive UI (Streamlit frontend)
├── evaluation/         # Dice, IoU, Precision, Recall, and Confusion Matrices
├── models/             # PyTorch Neural Network Definitions (U-Net)
├── semi_supervised/    # Pseudo-label generation pipeline
├── xai/                # Explainable AI (Grad-CAM & Attention maps)
├── utils/              # Preprocessing, augmentations, and dataset loaders
└── app.py              # Main dashboard entry point
```

## 🚀 Getting Started

### 1. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/Medical_Segmentation.git
cd Medical_Segmentation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Data Preparation

Place your medical dataset in the `dataset/` directory. The pipeline expects paired images and ground truth masks (if available). 

*(Note: Specific dataset conversion scripts for standard open-source datasets like LGG MRI are included in the repository.)*

### 3. Launching the Dashboard

The easiest way to interact with the framework is via the web dashboard.

```bash
streamlit run app.py
```

From the dashboard, you can:
- **Run Live Inference:** Upload a scan and instantly view the segmentation mask and XAI heatmaps.
- **Review Training:** Analyze loss curves and historical metrics.
- **Export Reports:** Generate Active Learning target lists for human annotators.

### 4. CLI Execution

For headless server environments, you can trigger individual pipeline stages directly:

```bash
# Train the model from scratch
python train.py

# Run standalone explainable AI diagnostics
python test_xai.py

# Execute the benchmarking suite
python -c "from benchmarking import run_benchmark; run_benchmark()"
```

## 🗺️ Future Roadmap

This framework is continuously evolving. Planned upgrades include:

- [ ] **3D Volumetric Segmentation**: Upgrading the backbone to a V-Net/3D U-Net to process full 3D MRI scans natively.
- [ ] **Multi-Class Detection**: Expanding the architecture to detect multiple tumor sub-regions (e.g., edema vs. enhancing core).
- [ ] **Advanced XAI (SHAP)**: Integrating SHapley Additive exPlanations for even deeper pixel-level accountability.
- [ ] **Federated Learning Support**: Enabling decentralized training across hospitals to preserve strict patient data privacy (HIPAA compliance).
- [ ] **Docker/FastAPI Deployment**: Containerizing the model for high-throughput, low-latency hospital API integration.

---

<div align="center">
  <i>Developed for advancing the intersection of Deep Learning and Medical Imaging.</i><br>
  <b>Harsh Khutela - Engineer and Research Intern at CSIR 4PI</b>
</div>
