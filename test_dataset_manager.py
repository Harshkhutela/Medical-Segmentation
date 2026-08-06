"""Test runner for the dataset management module."""

from dataset_manager import DatasetManager


def main() -> None:
    """Run the dataset manager and print the generated outputs."""
    print("Medical Dataset Manager\n")

    manager = DatasetManager()
    result = manager.analyze()

    print(f"Total Images : {result['total_images']}")
    print(f"Total Masks : {result['total_masks']}")
    print(f"Total Valid Pairs : {result['total_valid_pairs']}")
    print("Dataset Report Generated")
    print(result["report_csv"])
    print(result["summary_txt"])
    print(result["preview_png"])
    print("\nDataset Manager Completed Successfully")


if __name__ == "__main__":
    main()
