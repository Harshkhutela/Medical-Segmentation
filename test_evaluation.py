"""Run the experiment evaluation pipeline and print a compact summary."""

from configs.config import BASE_DIR, OUTPUT_PATH
from evaluation import ExperimentEvaluator, generate_experiment_plots


def main():
    """Load experiment artifacts, export CSVs, generate plots, and summarize."""
    print("-------------------------------------\n")
    print("Experiment Summary\n")

    evaluator = ExperimentEvaluator()
    evaluator.ensure_output_folders()

    history = evaluator.load_training_history()
    active_learning_results = evaluator.load_active_learning_results()
    semi_supervised_report = evaluator.load_semi_supervised_report()

    evaluator.export_training_csv(history)
    generate_experiment_plots(history, evaluator.plots_dir)

    summary = evaluator.compute_summary(
        history=history,
        active_learning_results=active_learning_results,
        semi_supervised_report=semi_supervised_report,
    )

    print(f"Best Dice : {summary['best_dice']:.4f}")
    print(f"Best IoU : {summary['best_iou']:.4f}")
    print(f"Best Epoch : {summary['best_epoch']}")
    print(f"Average Training Loss : {summary['average_training_loss']:.4f}")
    print(f"Final Epoch : {summary['final_epoch']}")
    print(f"Total Epochs : {summary['total_epochs']}")
    print(f"Pseudo Labels Generated : {summary['pseudo_labels_generated']}")
    print(
        "Most Uncertain Images Evaluated : "
        f"{summary['most_uncertain_images_evaluated']}"
    )
    print("\n-------------------------------------")

    print()
    print("CSV Saved")
    print((OUTPUT_PATH / "experiment_results.csv").relative_to(BASE_DIR))
    print("Plots Saved")
    print((OUTPUT_PATH / "plots").relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
