"""
Contact sheet viewer — shows a grid of random images from your augmented dataset.
Run this AFTER augment_dataset.py to sanity-check augmentation quality.
Opens the contact sheet image using Windows Photo Viewer.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUG_DIR      = PROJECT_ROOT / "data" / "augmented"
OUT_PATH     = PROJECT_ROOT / "data" / "contact_sheet.jpg"

GRID_COLS  = 6
GRID_ROWS  = 5
THUMB_SIZE = (200, 200)
SEED       = 42
random.seed(SEED)


def make_contact_sheet():
    all_images = list(AUG_DIR.rglob("*.jpg")) + list(AUG_DIR.rglob("*.png"))
    if not all_images:
        print(f"[ERROR] No images found in {AUG_DIR}")
        print("  → Run augment_dataset.py first!")
        return

    print(f"  Found {len(all_images)} augmented images across all classes.")

    sample = random.sample(all_images, min(GRID_COLS * GRID_ROWS, len(all_images)))
    rows = []

    for r in range(GRID_ROWS):
        row_imgs = []
        for c in range(GRID_COLS):
            idx = r * GRID_COLS + c
            if idx >= len(sample):
                tile = np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8)
            else:
                img = cv2.imread(str(sample[idx]))
                if img is None:
                    tile = np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8)
                else:
                    tile = cv2.resize(img, THUMB_SIZE)
                    cls_name = sample[idx].parent.name   # e.g. nozzle_1
                    cv2.putText(tile, cls_name, (4, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            row_imgs.append(tile)
        rows.append(np.hstack(row_imgs))

    sheet = np.vstack(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), sheet)
    print(f"  Contact sheet saved → {OUT_PATH}")

    # Open with Windows default image viewer (no GUI dependency needed)
    os.startfile(str(OUT_PATH))
    print("  Opened in Windows Photo Viewer ✓")


if __name__ == "__main__":
    make_contact_sheet()
