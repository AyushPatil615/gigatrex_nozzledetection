"""
Augment Anomaly / Unknown class only
======================================
Reads  : data/anomoly/*.jpeg  (bolt, nuts, washer, washer1)
Outputs: data/augmented/unknown/   (30 augmented copies per image)

Run:
  python data_pipeline/augment_anomaly.py
"""

import cv2
import numpy as np
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANOMALY_DIR  = PROJECT_ROOT / "data" / "anomoly"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "augmented" / "unknown"
COPIES       = 30
IMG_SIZE     = (640, 640)
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)


# ── Augmentation helpers (same as main script) ────────────────────────────────

def letterbox(img, size=(640, 640)):
    h, w   = img.shape[:2]
    scale  = min(size[0]/w, size[1]/h)
    nw, nh = int(w*scale), int(h*scale)
    img    = cv2.resize(img, (nw, nh))
    canvas = np.full((size[1], size[0], 3), 114, dtype=np.uint8)
    top    = (size[1]-nh)//2;  left = (size[0]-nw)//2
    canvas[top:top+nh, left:left+nw] = img
    return canvas

def augment(img):
    # Brightness / contrast
    b = random.randint(-45, 45);  c = random.uniform(0.6, 1.4)
    img = np.clip(img.astype(np.float32)*c+b, 0, 255).astype(np.uint8)
    # HSV jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] += random.uniform(-15,15); hsv[:,:,1] += random.uniform(-40,40)
    hsv[:,:,2] += random.uniform(-40,40)
    img = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Rotation (full 360)
    angle = random.uniform(0, 360)
    h, w  = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w,h), borderValue=(114,114,114))
    # Flip
    if random.random() > 0.4: img = cv2.flip(img, random.choice([-1,0,1]))
    # Scale crop
    if random.random() > 0.4:
        s = random.uniform(0.68, 1.0); nh2=int(h*s); nw2=int(w*s)
        t = random.randint(0,h-nh2); l = random.randint(0,w-nw2)
        img = cv2.resize(img[t:t+nh2, l:l+nw2], (w,h))
    # Blur
    if random.random() > 0.5:
        k = random.choice([3,5]); img = cv2.GaussianBlur(img,(k,k),0)
    # Noise
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(0,20), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    imgs = sorted(list(ANOMALY_DIR.glob("*.jpg")) +
                  list(ANOMALY_DIR.glob("*.jpeg")) +
                  list(ANOMALY_DIR.glob("*.png")))

    if not imgs:
        print(f"[ERROR] No images found in {ANOMALY_DIR}"); return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Anomaly Augmentation  →  class: unknown")
    print(f"  Input : {ANOMALY_DIR}  ({len(imgs)} images)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Copies per image: {COPIES}")
    print(f"{'='*55}\n")

    total = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name}"); continue

        img  = letterbox(img, IMG_SIZE)
        stem = img_path.stem

        # Original
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_orig.jpg"), img);  total += 1

        # Augmented copies
        for i in range(COPIES):
            aug = augment(img.copy())
            cv2.imwrite(str(OUTPUT_DIR / f"{stem}_aug{i:03d}.jpg"), aug)
            total += 1

        print(f"  ✓  {img_path.name:20s}  →  {COPIES+1} images")

    print(f"\n  DONE — {total} unknown/anomaly images saved")
    print(f"  → {OUTPUT_DIR.resolve()}")
    print(f"\n  Next: python auto_annotate.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()
