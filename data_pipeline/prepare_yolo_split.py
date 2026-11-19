"""
Prepare YOLO Detection Dataset Split
======================================
Splits data/annotated/ into train/val and writes data.yaml

Run:
  python data_pipeline/prepare_yolo_split.py
"""

import shutil, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANN_DIR      = PROJECT_ROOT / "data" / "annotated"
OUT_DIR      = PROJECT_ROOT / "data" / "yolo_dataset"
VAL_RATIO    = 0.15
SEED         = 42

CLASS_NAMES  = ["nozzle_1", "nozzle_2", "nozzle_3", "nozzle_4", "nozzle_5", "unknown"]

random.seed(SEED)


def prepare():
    img_dir = ANN_DIR / "images"
    lbl_dir = ANN_DIR / "labels"

    if not img_dir.exists():
        print(f"[ERROR] Not found: {img_dir}")
        print("  → Run auto_annotate.py first")
        return

    all_imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    paired   = [p for p in all_imgs if (lbl_dir / (p.stem + ".txt")).exists()]
    skipped  = len(all_imgs) - len(paired)

    if skipped:
        print(f"  [WARN] {skipped} images have no matching label — skipped")

    random.shuffle(paired)
    n_val    = max(1, int(len(paired) * VAL_RATIO))
    val_imgs = paired[:n_val]
    trn_imgs = paired[n_val:]

    for split, imgs in [("train", trn_imgs), ("val", val_imgs)]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_p in imgs:
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            shutil.copy2(img_p, OUT_DIR / "images" / split / img_p.name)
            shutil.copy2(lbl_p, OUT_DIR / "labels" / split / lbl_p.name)

    # Write data.yaml
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\n{'='*55}")
    print(f"  YOLO Dataset Split")
    print(f"  Train : {len(trn_imgs)}  |  Val : {len(val_imgs)}")
    print(f"  YAML  : {yaml_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    prepare()
