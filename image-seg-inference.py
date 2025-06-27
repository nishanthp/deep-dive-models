#!/usr/bin/env python3
"""
infer_human_seg.py
------------------
One-click inference for the U-Net human-segmentation model you trained.

Usage
-----
python infer_human_seg.py --image path/to/img.jpg [--weights unet_human_seg.pth] \
                          [--out mask.png] [--device cpu|cuda]

Output
------
• A binary mask (white = person, black = background) saved to --out.
"""

import argparse, numpy as np
from pathlib import Path
from PIL import Image
import torch
import albumentations as A
import segmentation_models_pytorch as smp

# ------------------------------------------------------------------------- #
# Argument parsing
# ------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image",   required=True,  help="input RGB image")
    p.add_argument("-w", "--weights", default="unet_human_seg.pth",
                   help="path to trained weight file")
    p.add_argument("-o", "--out",     default="mask.png",
                   help="output mask filename")
    p.add_argument("-d", "--device",  default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"], help="run on cpu or cuda")
    return p.parse_args()

# ------------------------------------------------------------------------- #
# Pre-process helper
# ------------------------------------------------------------------------- #
norm = A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def preprocess(img_pil: Image.Image):
    img = img_pil.convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img)
    arr = norm(image=arr)["image"]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
    return tensor  # [1,3,256,256]

# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. Load model
    model = smp.Unet("resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 2. Load & prep image
    img = Image.open(args.image)
    x = preprocess(img).to(device)

    # 3. Inference
    with torch.no_grad():
        logits = model(x)
        mask   = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255

    # 4. Save result
    Image.fromarray(mask).save(args.out)
    print(f"✓ Saved binary mask to {args.out}")

if __name__ == "__main__":
    main()

