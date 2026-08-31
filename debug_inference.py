"""Debug script to find a sample with foreground and test inference."""

import torch
from configs.config import CHECKPOINT_PATH, DEVICE
from models import UNet
from utils.dataset import MedicalSegmentationDataset

model = UNet(in_channels=3).to(DEVICE)
state = torch.load(CHECKPOINT_PATH / "best_model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

ds = MedicalSegmentationDataset()
print(f"Total samples: {len(ds)}")

found_fg = 0
for i in range(min(100, len(ds))):
    img, mask = ds[i]
    mask_fg = (mask > 0).float().sum().item()
    if mask_fg > 0:
        found_fg += 1
        inp = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)
            probs = torch.sigmoid(out)
            binary = (probs >= 0.5).float()
        
        pred_fg = int(binary.sum().item())
        print(f"Sample {i}: GT_fg={int(mask_fg)}, Pred_fg={pred_fg}, "
              f"Logit=[{out.min().item():.2f}, {out.max().item():.2f}], "
              f"Prob=[{probs.min().item():.6f}, {probs.max().item():.6f}]")
        
        if found_fg >= 10:
            break

if found_fg == 0:
    print("No foreground samples found in first 100 samples!")

# Also try with a lower threshold
print("\n--- Testing lower thresholds on first FG sample ---")
for i in range(min(len(ds), 500)):
    img, mask = ds[i]
    if (mask > 0).float().sum().item() > 0:
        inp = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)
            probs = torch.sigmoid(out)
        
        for thr in [0.5, 0.3, 0.1, 0.05, 0.01, 0.001]:
            fg = int((probs >= thr).float().sum().item())
            print(f"  threshold={thr}: foreground={fg}")
        
        print(f"  Prob stats: mean={probs.mean().item():.8f}, max={probs.max().item():.8f}")
        break
