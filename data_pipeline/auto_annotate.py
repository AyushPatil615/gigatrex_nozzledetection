"""
Auto-Annotate Augmented Nozzle Images
=======================================
Since nozzles are on plain white/grey backgrounds, we can automatically
find the bounding box using contour detection — no manual labeling needed!

Reads : data/augmented/<class>/*.jpg
Writes: data/annotated/images/*.jpg   +   data/annotated/labels/*.txt (YOLO format)

YOLO label format per line:
  class_id  cx  cy  w  h   (all normalized 0–1)

Run:
  python data_pipeline/auto_annotate.py
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
AUG_DIR       = PROJECT_ROOT / "data" / "augmented"
OUT_IMG_DIR   = PROJECT_ROOT / "data" / "annotated" / "images"
OUT_LBL_DIR   = PROJECT_ROOT / "data" / "annotated" / "labels"

# Class name → YOLO class index (must match data.yaml)
CLASS_INDEX = {
    "nozzle_1": 0,
    "nozzle_2": 1,
    "nozzle_3": 2,
    "nozzle_4": 3,
    "nozzle_5": 4,
    "unknown" : 5,   # nuts, bolts, washers — things that look similar but aren't nozzles
}

PADDING = 0.05   # 5% extra padding around detected bbox


def find_nozzle_bbox(img_bgr):
    """
    Find the nozzle bounding box using background subtraction.
    Works because nozzles are on plain white/grey backgrounds.
    Returns (cx, cy, w, h) normalized 0-1, or None if not found.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Try method 1: Otsu threshold (nozzle is darker than background) ---
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: invert (nozzle brighter than background)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour = nozzle
    cnt = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(cnt) / (w * h)

    # Reject if too small (<5%) or too large (>95%) — probably noise or whole image
    if area_ratio < 0.05 or area_ratio > 0.95:
        # Fallback bbox = center 80% of image
        return 0.5, 0.5, 0.80, 0.80

    x, y, bw, bh = cv2.boundingRect(cnt)

    # Add padding
    pad_x = int(bw * PADDING)
    pad_y = int(bh * PADDING)
    x1 = max(0, x - pad_x);     y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x); y2 = min(h, y + bh + pad_y)

    # Normalize to 0-1
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    nw = (x2 - x1) / w
    nh = (y2 - y1) / h

    return cx, cy, nw, nh


def run():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in AUG_DIR.iterdir() if d.is_dir()])
    if not classes:
        print(f"[ERROR] No class folders found in {AUG_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"  Auto-Annotation (background subtraction)")
    print(f"  Input : {AUG_DIR}")
    print(f"  Output: {OUT_IMG_DIR.parent}")
    print(f"{'='*60}\n")

    total = 0
    failed = 0

    for cls in classes:
        cls_id = CLASS_INDEX.get(cls)
        if cls_id is None:
            print(f"  [SKIP] Unknown class: {cls}")
            continue

        images = list((AUG_DIR / cls).glob("*.jpg")) + \
                 list((AUG_DIR / cls).glob("*.png"))

        cls_ok = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            bbox = find_nozzle_bbox(img)
            if bbox is None:
                failed += 1
                continue

            cx, cy, nw, nh = bbox

            # Copy image
            dest_img = OUT_IMG_DIR / img_path.name
            shutil.copy2(img_path, dest_img)

            # Write YOLO label
            dest_lbl = OUT_LBL_DIR / (img_path.stem + ".txt")
            with open(dest_lbl, "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            cls_ok += 1
            total  += 1

        print(f"  ✓ {cls:12s}  →  {cls_ok} images annotated")

    print(f"\n  DONE  →  {total} images annotated,  {failed} skipped")
    print(f"  Output: {OUT_IMG_DIR.parent.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
