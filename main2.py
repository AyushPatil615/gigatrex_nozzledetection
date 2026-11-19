"""
2-Stage Nozzle Detection Pipeline (fixed)
=========================================
Stage 1: Hough Circle detection -> finds all tray holes -> keeps the ones holding a nozzle
Stage 2: YOLOv8s classifier     -> classifies each nozzle crop

Usage:
    python main_fixed.py                    # default tray image
    python main_fixed.py <image.jpg>        # specific image
    python main_fixed.py --webcam           # live webcam
    python main_fixed.py --debug            # save Stage 1 debug image
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import sys
import os

# -- CONFIG -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "runs" / "nozzle_detect_v1" / "weights" / "best.pt"
OUTPUT_DIR   = PROJECT_ROOT / "detect"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES     = ["nozzle_1", "nozzle_2", "nozzle_3", "nozzle_4", "nozzle_5", "unknown"]
CONF_THRESH = 0.30
REVIEW_BAND = 0.60
CAMERA_ID   = 0
DEBUG       = "--debug" in sys.argv

# Stage 1 tuning (all relative to image width so it works at any resolution)
MIN_R_FRAC    = 0.018    # min hole radius  (~23 px @ 1280)
MAX_R_FRAC    = 0.036    # max hole radius  (~46 px @ 1280)
MIN_DIST_FRAC = 0.045    # min distance between hole centres (~58 px @ 1280)
HOUGH_PARAM1  = 80       # Canny high threshold
HOUGH_PARAM2  = 30       # accumulator threshold (lower = more circles)

BLUE_DIFF     = 40       # pixel counts as "empty tray blue" if (B - R) > this
EMPTY_RATIO   = 0.50     # if more than this fraction of the centre is blue -> empty

COLORS = {
    "nozzle_1": (180, 180, 180),
    "nozzle_2": (80,  200,  80),
    "nozzle_3": (255, 220, 100),
    "nozzle_4": (0,   190, 220),
    "nozzle_5": (200,  80, 200),
    "unknown" : (0,     0, 255),
}

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun: python train.py")
        _model = YOLO(str(MODEL_PATH))
    return _model


# -- STAGE 1: Locate nozzles using Hough Circles --------------------------------

def hole_blue_ratio(img_bgr, cx, cy, r):
    """Fraction of the hole centre that is strongly blue (tray colour)."""
    h, w = img_bgr.shape[:2]
    pad = int(r * 0.6)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)
    crop = img_bgr[y1:y2, x1:x2].astype(np.int16)
    if crop.size == 0:
        return 1.0
    blue_dom = (crop[:, :, 0] - crop[:, :, 2]) > BLUE_DIFF   # B much greater than R
    return float(blue_dom.mean())


def detect_nozzle_regions(img_bgr, debug=False):
    """
    Finds circular tray holes with Hough, then keeps those that are NOT
    empty (empty = mostly blue tray visible through the hole).

    Returns list of (x1, y1, x2, y2) pixel bboxes.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    min_r    = int(w * MIN_R_FRAC)
    max_r    = int(w * MAX_R_FRAC)
    min_dist = int(w * MIN_DIST_FRAC)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    boxes = []
    debug_img = img_bgr.copy() if debug else None

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for (cx, cy, r) in circles:
            ratio    = hole_blue_ratio(img_bgr, cx, cy, r)
            is_empty = ratio > EMPTY_RATIO

            if debug_img is not None:
                col = (0, 0, 200) if is_empty else (0, 200, 0)
                cv2.circle(debug_img, (cx, cy), r, col, 2)
                cv2.putText(debug_img, f"{ratio:.2f}", (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

            if not is_empty:
                pad2 = int(r * 1.15)
                bx1 = max(0, cx - pad2); by1 = max(0, cy - pad2)
                bx2 = min(w, cx + pad2); by2 = min(h, cy + pad2)
                boxes.append((bx1, by1, bx2, by2))

    if debug_img is not None:
        out = OUTPUT_DIR / "debug_circles.jpg"
        cv2.imwrite(str(out), debug_img)
        try:
            os.startfile(str(out))
        except Exception:
            pass
        print(f"[DEBUG] Hough circles saved -> {out}")
        print(f"[DEBUG] Radius range        : {min_r}-{max_r} px")
        print(f"[DEBUG] Total circles found : {0 if circles is None else len(circles)}")
        print(f"[DEBUG] Nozzle candidates   : {len(boxes)}")

    return boxes


# -- STAGE 2: Classify each crop ------------------------------------------------

def classify_crop(model, crop_bgr):
    result = model(crop_bgr, verbose=False)[0]
    probs  = result.probs
    cls_id = int(probs.top1)
    conf   = float(probs.top1conf)
    label  = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
    return label, conf


# -- COMBINED PIPELINE ----------------------------------------------------------

def detect_and_classify(img_bgr, debug=False):
    t0    = time.time()
    model = get_model()

    boxes        = detect_nozzle_regions(img_bgr, debug=debug)
    detections   = []
    class_counts = {c: 0 for c in CLASSES}
    annotated    = img_bgr.copy()

    for (x1, y1, x2, y2) in boxes:
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, conf = classify_crop(model, crop)
        if conf < CONF_THRESH:
            label = "unknown"

        color        = COLORS.get(label, (128, 128, 128))
        needs_review = conf < REVIEW_BAND

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        flag = " ?" if needs_review else ""
        tag  = f"{label} {conf:.2f}{flag}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, tag, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "label": label, "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2], "needs_review": needs_review
        })
        class_counts[label] = class_counts.get(label, 0) + 1

    latency = (time.time() - t0) * 1000

    nozzle_total = sum(v for k, v in class_counts.items() if k != "unknown")
    hud = [
        f"Nozzles: {nozzle_total}  Unknown: {class_counts.get('unknown', 0)}  Latency: {latency:.0f}ms",
    ] + [f"  {k}: {v}" for k, v in class_counts.items() if v > 0]

    for i, line in enumerate(hud):
        y = 25 + i * 22
        cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated, detections, class_counts, latency


# -- IMAGE MODE -----------------------------------------------------------------

def predict(image_path: str, show=True, debug=False) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    annotated, detections, class_counts, latency = detect_and_classify(img, debug=debug)

    out_path = OUTPUT_DIR / f"{Path(image_path).stem}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    if show:
        try:
            os.startfile(str(out_path))
        except Exception:
            pass

    return {
        "total_detected": len(detections),
        "class_counts"  : {k: v for k, v in class_counts.items() if v > 0},
        "needs_review"  : any(d["needs_review"] for d in detections),
        "latency_ms"    : round(latency, 1),
        "annotated_path": str(out_path),
        "detections"    : detections,
    }


# -- WEBCAM MODE ----------------------------------------------------------------

def run_webcam():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_ID}")
        return

    print("\n  Live Detection | Q: Quit  S: Save  P: Pause\n")
    paused = False
    snap = 0
    annotated = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                continue
            annotated, _, _, _ = detect_and_classify(frame)
            cv2.putText(annotated, "Q:Quit S:Save P:Pause",
                        (8, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if annotated is not None:
            cv2.imshow("Lechler Nozzle Detection", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and annotated is not None:
            p = OUTPUT_DIR / f"snap_{snap:04d}.jpg"
            cv2.imwrite(str(p), annotated)
            snap += 1
            print(f"  Saved -> {p}")
        elif key == ord('p'):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'RESUMED'}")

    cap.release()
    cv2.destroyAllWindows()


# -- ENTRY POINT ----------------------------------------------------------------

if __name__ == "__main__":
    if "--webcam" in sys.argv:
        run_webcam()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if args:
            img_path = args[0]
        else:
            candidates = list((PROJECT_ROOT / "data").glob("WIN_*.jpg"))
            if not candidates:
                print("Usage: python main_fixed.py <image.jpg>  |  --webcam  |  --debug")
                sys.exit(1)
            img_path = str(candidates[0])
            print(f"[INFO] Using: {img_path}\n")

        result = predict(img_path, show=True, debug=DEBUG)

        print(f"\n{'=' * 55}")
        print(f"  NOZZLE DETECTION RESULTS")
        print(f"{'=' * 55}")
        print(f"  Total Detected : {result['total_detected']}")
        print(f"  Latency        : {result['latency_ms']} ms")
        print(f"  Needs Review?  : {'YES' if result['needs_review'] else 'NO'}")
        print(f"\n  Counts by Type:")
        for k, v in result["class_counts"].items():
            print(f"    {k:12s}: {v}  {'#' * v}")
        print(f"\n  Saved -> {result['annotated_path']}")
        print(f"{'=' * 55}\n")