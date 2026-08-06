"""Smoke test for the explainable AI pipeline."""

from __future__ import annotations

import sys

from configs.config import BASE_DIR
from xai.utils import load_first_dataset_image, load_xai_model, run_xai_pipeline


def main() -> int:
    """Run the complete XAI pipeline and verify saved artifacts."""
    print("Explainable AI")

    try:
        image_path, _ = load_first_dataset_image()
    except Exception as error:
        print(f"Error: unable to load a dataset sample. {error}")
        return 1

    print("Loading Checkpoint...")
    try:
        model = load_xai_model()
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return 1

    print("Running XAI Pipeline...")
    try:
        result = run_xai_pipeline(image_path, model=model)
    except Exception as error:
        print(f"Error: XAI pipeline failed. {error}")
        return 1

    required_paths = [
        result.output_paths["heatmap"],
        result.output_paths["overlay"],
        result.output_paths["attention"],
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        print("Error: some XAI artifacts were not saved correctly.")
        for path in missing_paths:
            print(f"Missing: {path}")
        return 1

    print("Prediction Complete")
    print(f"Image Name : {result.image_name}")
    print(f"Prediction Confidence : {result.confidence:.4f}")
    print(f"Highlighted Region : {result.highlighted_region}")
    print("XAI Artifacts Saved")
    for path in required_paths:
        print(path.relative_to(BASE_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
