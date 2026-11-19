"""
Prepare Classification Dataset Split for YOLOv8
=================================================
Takes data/augmented/<class>/ and splits into train/val.

Output structure (what YOLOv8 classify expects):
  data/classify/
    train/
      nozzle_1/   nozzle_2/   nozzle_3/   nozzle_4/   nozzle_5/
    val/
      nozzle_1/   nozzle_2/   nozzle_3/   nozzle_4/   nozzle_5/
"""

import shutil, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUG_DIR      = PROJECT_ROOT / "data" / "augmented"
OUT_DIR      = PROJECT_ROOT / "data" / "classify"
VAL_RATIO    = 0.15
SEED         = 42

random.seed(SEED)

def prepare():
    classes = sorted([d.name for d in AUG_DIR.iterdir() if d.is_dir()])
    if not classes:
        print(f"[ERROR] No class folders in {AUG_DIR}")
        return

    print(f"\n{'='*55}")
    print(f"  Classification Split")
    print(f"  Classes : {classes}")
    print(f"  Val     : {int(VAL_RATIO*100)}%")
    print(f"{'='*55}\n")

    for cls in classes:
        images = list((AUG_DIR / cls).glob("*.jpg")) + \
                 list((AUG_DIR / cls).glob("*.png"))
        random.shuffle(images)

        n_val   = max(1, int(len(images) * VAL_RATIO))
        val_imgs = images[:n_val]
        trn_imgs = images[n_val:]

        for split, imgs in [("train", trn_imgs), ("val", val_imgs)]:
            dest = OUT_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {cls}: {len(trn_imgs)} train | {len(val_imgs)} val")

    total_train = sum(len(list((OUT_DIR/"train"/c).glob("*"))) for c in classes)
    total_val   = sum(len(list((OUT_DIR/"val"/c).glob("*")))   for c in classes)
    print(f"\n  TOTAL → train: {total_train}  val: {total_val}")
    print(f"  Output: {OUT_DIR.resolve()}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    prepare()
