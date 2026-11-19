"""
Nozzle Augmentation Pipeline
==============================
Reads images from data/images/ and maps them by filename prefix:
  typeA* → nozzle_1  (smooth silver ball / dome)
  typeB* → nozzle_2  (perforated silver ball)
  typeC* → nozzle_3  (gold flat-top with slots)
  typeD* → nozzle_4  (gold spiral / screw type)
  typeE* → nozzle_5  (silver flat disc / washer)

Output → data/augmented/<class>/   (30 augmented + 1 original per raw image)

Run:
  pip install opencv-python numpy
  python data_pipeline/augment_dataset.py
"""

import cv2
import numpy as np
from pathlib import Path
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Always works regardless of which folder you run from
PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # lechler_detect/
INPUT_DIR      = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "augmented"
IMAGES_PER_RAW = 30          # augmented copies per source image
IMG_SIZE       = (640, 640)
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)

# ── FILENAME PREFIX → CLASS NAME ─────────────────────────────────────────────
PREFIX_MAP = {
    "typea": "nozzle_1",   # smooth silver ball / dome, top view
    "typeb": "nozzle_2",   # perforated silver ball
    "typec": "nozzle_3",   # gold flat-top with arrow slots
    "typed": "nozzle_4",   # gold spiral / conical screw type
    "typee": "nozzle_5",   # silver flat disc / washer with center hole
}

# ── ANOMALY / UNKNOWN CLASS — separate input folder ──────────────────────────
ANOMALY_DIR = PROJECT_ROOT / "data" / "anomoly"   # your folder name (kept as-is)
ANOMALY_CLS = "unknown"

# ── AUGMENTATION HELPERS ──────────────────────────────────────────────────────

def letterbox(img, size=(640, 640), pad_color=(114, 114, 114)):
    """Resize preserving aspect ratio, pad to square."""
    h, w   = img.shape[:2]
    scale  = min(size[0] / w, size[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    img    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size[1], size[0], 3), pad_color, dtype=np.uint8)
    top    = (size[1] - nh) // 2
    left   = (size[0] - nw) // 2
    canvas[top:top+nh, left:left+nw] = img
    return canvas


def rand_brightness_contrast(img):
    b   = random.randint(-45, 45)
    c   = random.uniform(0.60, 1.40)
    img = np.clip(img.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
    return img


def rand_hsv(img, h=15, s=40, v=40):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] += random.uniform(-h, h)
    hsv[:, :, 1] += random.uniform(-s, s)
    hsv[:, :, 2] += random.uniform(-v, v)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rand_rotate(img):
    """Full 360° — nozzles viewed from above are rotationally symmetric."""
    angle = random.uniform(0, 360)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))


def rand_flip(img):
    return cv2.flip(img, random.choice([-1, 0, 1]))


def rand_scale_crop(img, lo=0.68, hi=1.0):
    s    = random.uniform(lo, hi)
    h, w = img.shape[:2]
    nh   = int(h * s);  nw = int(w * s)
    top  = random.randint(0, h - nh)
    left = random.randint(0, w - nw)
    return cv2.resize(img[top:top+nh, left:left+nw], (w, h))


def rand_perspective(img, d=0.07):
    h, w  = img.shape[:2]
    pts1  = np.float32([[0,0],[w,0],[0,h],[w,h]])
    delta = np.random.uniform(-d*min(w,h), d*min(w,h), pts1.shape).astype(np.float32)
    M     = cv2.getPerspectiveTransform(pts1, pts1 + delta)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(114, 114, 114))


def rand_shadow(img):
    """Simulate one-sided shadow / uneven lighting."""
    out  = img.astype(np.float32)
    h, w = img.shape[:2]
    alpha = random.uniform(0.25, 0.65)
    side  = random.choice(['L', 'R', 'T', 'B'])
    split = random.randint(w//4, 3*w//4) if side in ('L','R') else \
            random.randint(h//4, 3*h//4)
    if   side == 'L': out[:, :split]  *= alpha
    elif side == 'R': out[:, split:]  *= alpha
    elif side == 'T': out[:split, :]  *= alpha
    else:             out[split:, :]  *= alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def rand_blur(img):
    k = random.choice([1, 3, 5])
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


def rand_noise(img, sigma_max=20):
    sigma = random.uniform(0, sigma_max)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment(img):
    img = rand_brightness_contrast(img)
    img = rand_hsv(img)
    img = rand_rotate(img)
    if random.random() > 0.4: img = rand_flip(img)
    if random.random() > 0.4: img = rand_scale_crop(img)
    if random.random() > 0.4: img = rand_perspective(img)
    if random.random() > 0.5: img = rand_shadow(img)
    if random.random() > 0.5: img = rand_blur(img)
    if random.random() > 0.5: img = rand_noise(img)
    return img


# ── CLASS RESOLVER ────────────────────────────────────────────────────────────

def resolve_class(filename: str):
    """
    Return the nozzle class name from the filename prefix.
    typeA*.jpeg → nozzle_1, typeB* → nozzle_2 … etc.
    """
    lower = filename.lower()
    for prefix, cls in PREFIX_MAP.items():
        if lower.startswith(prefix):
            return cls
    return None


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder not found: {INPUT_DIR.resolve()}")
        return

    all_imgs = sorted(
        list(INPUT_DIR.glob("*.jpg"))  +
        list(INPUT_DIR.glob("*.jpeg")) +
        list(INPUT_DIR.glob("*.png"))
    )

    if not all_imgs:
        print(f"[ERROR] No images found in {INPUT_DIR.resolve()}")
        return

    print(f"\n{'='*62}")
    print(f"  Lechler Nozzle Augmentation Pipeline")
    print(f"  Input : {INPUT_DIR.resolve()}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"  Augmented copies per image : {IMAGES_PER_RAW}")
    print(f"  Images found : {len(all_imgs)}")
    print(f"{'='*62}\n")

    class_totals = {}
    skipped      = []

    for img_path in all_imgs:
        cls = resolve_class(img_path.name)
        if cls is None:
            skipped.append(img_path.name)
            print(f"  [SKIP] {img_path.name}  (no prefix match — add to PREFIX_MAP if needed)")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name}")
            continue

        img     = letterbox(img, IMG_SIZE)
        out_dir = OUTPUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        stem  = img_path.stem          # e.g. "typeA_side"
        saved = 0

        # 1. Save the original (letterboxed, no augmentation)
        cv2.imwrite(str(out_dir / f"{stem}_orig.jpg"), img)
        saved += 1

        # 2. Augmented copies
        for i in range(IMAGES_PER_RAW):
            aug = augment(img.copy())
            cv2.imwrite(str(out_dir / f"{stem}_aug{i:03d}.jpg"), aug)
            saved += 1

        class_totals[cls] = class_totals.get(cls, 0) + saved
        print(f"  ✓  {img_path.name:25s}  →  {cls}  ({saved} images written)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  AUGMENTATION COMPLETE\n")
    total = 0
    for cls in ["nozzle_1","nozzle_2","nozzle_3","nozzle_4","nozzle_5","unknown"]:
        n   = class_totals.get(cls, 0)
        bar = "█" * (n // 10)
        print(f"    {cls}:  {n:4d} images   {bar}")
        total += n
    print(f"    {'─'*42}")
    print(f"    TOTAL :  {total} images")

    if skipped:
        print(f"\n  [WARN] Skipped (unrecognised prefix): {skipped}")

    # ── ANOMALY / UNKNOWN class ───────────────────────────────────────────────
    if ANOMALY_DIR.exists():
        anomaly_imgs = sorted(
            list(ANOMALY_DIR.glob("*.jpg"))  +
            list(ANOMALY_DIR.glob("*.jpeg")) +
            list(ANOMALY_DIR.glob("*.png"))
        )
        if anomaly_imgs:
            print(f"\n  Processing anomaly/unknown class ({len(anomaly_imgs)} images)...")
            out_dir = OUTPUT_DIR / ANOMALY_CLS
            out_dir.mkdir(parents=True, exist_ok=True)
            anom_total = 0
            for img_path in anomaly_imgs:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img  = letterbox(img, IMG_SIZE)
                stem = img_path.stem
                cv2.imwrite(str(out_dir / f"{stem}_orig.jpg"), img)
                anom_total += 1
                for i in range(IMAGES_PER_RAW):
                    aug = augment(img.copy())
                    cv2.imwrite(str(out_dir / f"{stem}_aug{i:03d}.jpg"), aug)
                    anom_total += 1
            class_totals[ANOMALY_CLS] = anom_total
            print(f"  ✓  anomoly/           →  {ANOMALY_CLS}  ({anom_total} images written)")
    else:
        print(f"\n  [INFO] No anomaly folder found at {ANOMALY_DIR} — skipping.")

    print(f"""
  Next steps:
    1. Visually check  →  python data_pipeline/inspect_augmented.py
    2. Annotate        →  labelImg  (save as YOLO format)
    3. Split dataset   →  python data_pipeline/prepare_yolo_split.py
    4. Train           →  python train.py
{'='*62}
""")


if __name__ == "__main__":
    run()
