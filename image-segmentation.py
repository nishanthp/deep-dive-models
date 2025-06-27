"""
human_segmentation_unet.py
End-to-end training & inference for human foreground segmentation
===============================================================

Run:
    python human_segmentation_unet.py          # trains 10 epochs, saves weights
    python - <<'PY'
    from human_segmentation_unet import segment
    segment("my_photo.jpg")                    # writes mask.png
    PY
"""

import os, zipfile, urllib.request, random, numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# -----------------------------------------------------------------------------#
# 1 . Dataset download / layout
# -----------------------------------------------------------------------------#
DATA_URL = (
    "https://github.com/VikramShenoy97/"
    "Human-Segmentation-Dataset/archive/refs/heads/master.zip"
)
ROOT = Path("human_seg")
if not ROOT.exists():
    print("Downloading dataset …")
    zip_path = "human_seg.zip"
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall()
    os.remove(zip_path)
    os.rename("Human-Segmentation-Dataset-master", ROOT)

IMG_DIR  = ROOT / "Training_Images"
MASK_DIR = ROOT / "Ground_Truth"

# keep only pairs that really exist
def has_mask(img_name: str) -> bool:
    stem = Path(img_name).stem
    return (MASK_DIR / f"{stem}.png").exists() or (MASK_DIR / f"{stem}.jpg").exists()

all_images = sorted([p.name for p in IMG_DIR.glob("*.jpg") if has_mask(p.name)])
random.seed(42); random.shuffle(all_images)
val_size = int(0.15 * len(all_images))
val_imgs, train_imgs = all_images[:val_size], all_images[val_size:]

# -----------------------------------------------------------------------------#
# 2 . Dataset class   (★ includes every alignment / shape fix)
# -----------------------------------------------------------------------------#
class HumanSegDataset(Dataset):
    def __init__(self, names, img_dir, mask_dir, transform):
        self.names, self.img_dir, self.mask_dir, self.t = names, img_dir, mask_dir, transform

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        stem = Path(name).stem
        img_path  = self.img_dir / name
        mask_path = self.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = self.mask_dir / f"{stem}.jpg"

        # load RGB image + binary mask
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # ---- guarantee identical width/height BEFORE Albumentations ----
        target_size = (256, 256)  # (w, h)
        img  = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

        img  = np.array(img)
        mask = np.array(mask, dtype=np.float32) / 255.0          # [H,W] values 0/1

        # Albumentations expects mask shape [H,W]; after ToTensorV2 we add channel
        sample = self.t(image=img, mask=mask)
        mask_tensor = sample["mask"].unsqueeze(0)                # -> [1,H,W]
        return sample["image"], mask_tensor

# -----------------------------------------------------------------------------#
# 3 . Augmentations & DataLoaders
# -----------------------------------------------------------------------------#
train_tf = A.Compose([
    A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.5),
    A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ToTensorV2()
])
val_tf = A.Compose([
    A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ToTensorV2()
])

train_ds = HumanSegDataset(train_imgs, IMG_DIR, MASK_DIR, train_tf)
val_ds   = HumanSegDataset(val_imgs,   IMG_DIR, MASK_DIR, val_tf)

# num_workers=0 is safest across OSes (esp. macOS)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

# -----------------------------------------------------------------------------#
# 4 . Model, loss, optimiser
# -----------------------------------------------------------------------------#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)

dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce  = nn.BCEWithLogitsLoss()
def loss_fn(pred, target): return 0.5 * dice(pred, target) + 0.5 * bce(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# -----------------------------------------------------------------------------#
# 5 . Train / eval loops
# -----------------------------------------------------------------------------#
def train_epoch(loader):
    model.train(); running = 0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward(); optimizer.step()
        running += loss.item()
    return running / len(loader)

@torch.no_grad()
def evaluate(loader):
    model.eval(); iou_sum, n = 0, 0
    for x, y in tqdm(loader, desc="Val"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = torch.sigmoid(model(x)) > 0.5
        inter = (pred & (y > 0.5)).float().sum((1,2,3))
        union = (pred | (y > 0.5)).float().sum((1,2,3))
        iou_sum += ((inter + 1e-6) / (union + 1e-6)).sum().item()
        n += x.size(0)
    return iou_sum / n

# -----------------------------------------------------------------------------#
# 6 . Run training
# -----------------------------------------------------------------------------#
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_epoch(train_loader)
    val_iou = evaluate(val_loader)
    print(f"[{epoch:02d}/{EPOCHS}] loss: {tr_loss:.4f} | mIoU: {val_iou:.4f}")

torch.save(model.state_dict(), "unet_human_seg.pth")
print("✔ Model saved to unet_human_seg.pth")

# -----------------------------------------------------------------------------#
# 7 . Inference helper
# -----------------------------------------------------------------------------#
def segment(path_img: str, weights="unet_human_seg.pth"):
    raw  = Image.open(path_img).convert("RGB").resize((256,256), Image.BILINEAR)
    arr  = np.array(raw)
    arr  = A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))(image=arr)["image"]
    x    = torch.from_numpy(arr).permute(2,0,1).float().unsqueeze(0).to(DEVICE)

    net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    net.load_state_dict(torch.load(weights, map_location=DEVICE)); net.eval()
    with torch.no_grad():
        mask = (torch.sigmoid(net(x))[0,0] > 0.5).cpu().numpy().astype(np.uint8)*255
    Image.fromarray(mask).save("mask.png")
    print("✔ Saved binary mask as mask.png")

