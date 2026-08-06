"""Generate pseudo labels and prepare a semi-supervised retraining dataset."""

from configs.config import BASE_DIR, CHECKPOINT_PATH, OUTPUT_PATH
from semi_supervised.pseudo_label import PseudoLabelGenerator
from semi_supervised.retrain import prepare_retraining_dataset


def save_report(statistics, report_file):
    """Write the semi-supervised dataset summary to a plain-text report."""
    report = (
        "Semi-Supervised Learning Report\n"
        f"Original Labeled Images : {statistics['original_count']}\n"
        f"Pseudo Labeled Images : {statistics['pseudo_count']}\n"
        f"Final Dataset Size : {statistics['final_count']}\n"
        f"Generated Pseudo Labels : {statistics['generated']}\n"
        f"Skipped Images : {statistics['skipped']}\n"
        f"Average Confidence : {statistics['average_confidence']:.4f}\n"
        f"Retraining Dataset : {statistics['dataset_path']}\n"
    )
    report_file.write_text(report, encoding="utf-8")


def main():
    """Run pseudo-label generation, create retraining data, and save a report."""
    print("Semi-Supervised Learning\n")
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    if not checkpoint_file.is_file():
        print(
            "Error: trained checkpoint was not found. "
            "Run train.py before semi-supervised learning."
        )
        return

    generator = PseudoLabelGenerator(checkpoint_file=checkpoint_file)
    print("Loading Model...")
    try:
        generator.load_model()
    except (RuntimeError, OSError) as error:
        print(f"Error: unable to load checkpoint. {error}")
        return

    print("Generating Pseudo Labels...")
    pseudo_statistics = generator.generate_pseudo_labels()
    print(f"Generated : {pseudo_statistics['generated']}")
    print(f"Skipped : {pseudo_statistics['skipped']}")
    print(
        "Average Confidence : "
        f"{pseudo_statistics['average_confidence']:.4f}\n"
    )

    print("Preparing Retraining Dataset...")
    retraining_statistics = prepare_retraining_dataset()
    statistics = {**pseudo_statistics, **retraining_statistics}
    print(f"Original Labeled Images : {statistics['original_count']}")
    print(f"Pseudo Labeled Images : {statistics['pseudo_count']}")
    print(f"Final Dataset Size : {statistics['final_count']}\n")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_PATH / "semi_supervised_report.txt"
    save_report(statistics, report_file)
    print("Report Saved")
    print(report_file.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
