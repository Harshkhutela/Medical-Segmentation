"""Run U-Net inference on one dataset image and save a visual comparison."""

import torch

from configs.config import BASE_DIR, CHANNELS, CHECKPOINT_PATH, DEVICE, OUTPUT_PATH
from models import UNet
from utils.dataset import MedicalSegmentationDataset
from utils.metrics import dice_score, iou_score
from utils.visualize import save_prediction_visualization


def load_model(checkpoint_file):
    """Create U-Net and load trained weights from the supplied checkpoint path."""
    model = UNet(in_channels=CHANNELS).to(DEVICE)
    state_dict = torch.load(
        checkpoint_file,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    """Load one sample, predict its mask, score it, and save a visualization."""
    print("Medical Image Segmentation\n")
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"

    if not checkpoint_file.is_file():
        print(
            "Error: trained checkpoint was not found at "
            f"{checkpoint_file}. Run train.py before inference."
        )
        return

    print("Loading Checkpoint...")
    try:
        model = load_model(checkpoint_file)
    except (RuntimeError, OSError) as error:
        print(f"Error: unable to load checkpoint. {error}")
        return
    print("Checkpoint Loaded\n")

    dataset = MedicalSegmentationDataset()
    if len(dataset) == 0:
        print(
            "Error: no matching image-mask pairs were found. "
            "Check the dataset paths in configs/config.py."
        )
        return

    # The first paired sample supplies one RGB image and its ground-truth mask.
    image, ground_truth_mask = dataset[0]
    image_path, _ = dataset.samples[0]
    image_name = image_path.name

    print("Running Prediction...")
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(DEVICE)
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        predicted_mask = (probabilities >= 0.5).float()
    print("Prediction Complete\n")

    # Existing metric functions accept raw logits and the dataset's mask tensor.
    dice = dice_score(logits, ground_truth_mask.unsqueeze(0).to(DEVICE))
    iou = iou_score(logits, ground_truth_mask.unsqueeze(0).to(DEVICE))

    output_directory = OUTPUT_PATH / "predictions"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / "prediction_result.png"

    # Convert PyTorch channel-first tensors into Matplotlib-friendly arrays.
    image_array = image.permute(1, 2, 0).cpu().numpy()
    ground_truth_array = (ground_truth_mask > 0).cpu().numpy()
    prediction_array = predicted_mask.squeeze().cpu().numpy()
    save_prediction_visualization(
        image_array,
        ground_truth_array,
        prediction_array,
        output_file,
    )

    print(f"Image Name : {image_name}")
    print(f"Prediction Shape : {predicted_mask.shape}")
    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score : {iou:.4f}")
    print("Prediction Saved")
    print(output_file.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
