"""Run uncertainty-based active learning scoring on unlabeled images."""

import csv

from active_learning.query_strategy import select_top_k
from active_learning.sampler import ActiveSampler
from configs.config import BASE_DIR, CHECKPOINT_PATH, OUTPUT_PATH


def save_scores_csv(scores, csv_file):
    """Save all uncertainty scores to a CSV file for annotation planning."""
    fieldnames = [
        "filename",
        "least_confidence",
        "entropy",
        "margin",
        "final_score",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)


def print_scores_table(scores):
    """Print a readable uncertainty table for every scored image."""
    header = (
        f"{'Filename':<30} {'Least Confidence':>18} {'Entropy':>12} "
        f"{'Margin':>12} {'Final Score':>14}"
    )
    print(header)
    print("-" * len(header))
    for score in scores:
        print(
            f"{score['filename']:<30} "
            f"{score['least_confidence']:>18.4f} "
            f"{score['entropy']:>12.4f} "
            f"{score['margin']:>12.4f} "
            f"{score['final_score']:>14.4f}"
        )


def main():
    """Load U-Net, score unlabeled images, print rankings, and save a CSV."""
    print("Medical Active Learning\n")
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    if not checkpoint_file.is_file():
        print(
            "Error: trained checkpoint was not found. "
            "Run train.py before active learning."
        )
        return

    sampler = ActiveSampler(checkpoint_file=checkpoint_file)
    image_paths = sampler.find_unlabeled_images()
    if not image_paths:
        print("No unlabeled images found.")
        print("Please add images to dataset/unlabeled.")
        return

    print("Loading Model...")
    try:
        sampler.load_model()
    except (RuntimeError, OSError) as error:
        print(f"Error: unable to load checkpoint. {error}")
        return

    print("Scanning Unlabeled Dataset...")
    print(f"Found {len(image_paths)} images\n")
    print("Computing Uncertainty...")
    scores = sampler.score_unlabeled_images()
    print("Completed\n")

    print_scores_table(scores)
    top_images = select_top_k(scores, k=5)
    print("\nTop 5 Images")
    for score in top_images:
        print(score["filename"])

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT_PATH / "active_learning_scores.csv"
    save_scores_csv(scores, csv_file)
    print("\nCSV Saved")
    print(csv_file.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
