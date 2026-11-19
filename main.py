"""Detect and classify occupied nozzle tray slots with the trained YOLO detector.

Usage:
    python main.py [image.jpg] [--debug] [--no-show]
    python main.py --webcam
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import sys
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
_v2 = PROJECT_ROOT / "runs" / "nozzle_detect_v2" / "weights" / "best.pt"
_v1 = PROJECT_ROOT / "runs" / "nozzle_detect_v1" / "weights" / "best.pt"
MODEL_PATH   = _v2 if _v2.exists() else _v1   # auto-pick v2 when training finishes
OUTPUT_DIR   = PROJECT_ROOT / "detect"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES     = ["nozzle_1", "nozzle_2", "nozzle_3", "nozzle_4", "nozzle_5"]
CONF_THRESH = 0.20    # tuned for tray crops
REVIEW_BAND = 0.50    # flag anything below 50% with "?"
CROP_SIZE = 640
BATCH_SIZE = 8
EMPTY_RATIO = 0.48
DEBUG       = "--debug" in sys.argv

COLORS = {
    "nozzle_1": (180, 180, 180),
    "nozzle_2": (80,  200,  80),
    "nozzle_3": (255, 220, 100),
    "nozzle_4": (0,   190, 220),
    "nozzle_5": (200,  80, 200),
    "undetected": (100, 100, 100),
}

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun: python train.py")
        _model = YOLO(str(MODEL_PATH))
    return _model


# ── STAGE 1: Locate nozzles using Hough Circles ───────────────────────────────

def detect_nozzle_regions(img_bgr, debug=False):
    """Return occupied circles as (center_x, center_y, radius)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (7, 7), 1.5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, round(w * 0.044)),
        param1=75,
        param2=30,
        minRadius=max(1, round(w * 0.018)),
        maxRadius=max(2, round(w * 0.034)),
    )
    if circles is None:
        return []

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    candidates = []
    empty_centers = []
    for cx, cy, radius in np.rint(circles[0]).astype(int).tolist():
        pad = max(1, round(radius * 0.48))
        center = hsv[max(0, cy - pad):min(h, cy + pad + 1),
                     max(0, cx - pad):min(w, cx + pad + 1)]
        hue, saturation, value = np.moveaxis(center, -1, 0)
        empty_pixels = (value < 72) | ((hue >= 91) & (hue <= 140) & (saturation >= 55))
        ratio = float(empty_pixels.mean())
        candidates.append((cx, cy, radius, ratio))
        if ratio >= EMPTY_RATIO:
            empty_centers.append((cx, cy))

    # Empty slots define the tray interior. Allow for occupied outer rows.
    hull = None
    if len(empty_centers) >= 8:
        hull = cv2.convexHull(np.array(empty_centers, dtype=np.int32))

    occupied = []
    debug_img = img_bgr.copy() if debug else None
    for cx, cy, radius, ratio in candidates:
        in_tray = hull is None or cv2.pointPolygonTest(hull, (cx, cy), True) >= -w * 0.055
        accepted = ratio < EMPTY_RATIO and in_tray
        if accepted:
            occupied.append((cx, cy, radius))
        if debug_img is not None:
            color = (0, 180, 0) if accepted else (0, 0, 220)
            cv2.circle(debug_img, (cx, cy), radius, color, 2)
            cv2.putText(debug_img, f"{ratio:.2f}", (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if debug_img is not None:
        out = OUTPUT_DIR / "debug_circles.jpg"
        cv2.imwrite(str(out), debug_img)
        print(f"[DEBUG] {len(occupied)} occupied candidates from {len(candidates)} circles. Saved {out}")

    return sorted(occupied, key=lambda circle: (circle[1], circle[0]))


def prepare_crop(img_bgr, cx, cy, radius):
    """Extract natural nozzle crop with padding matching training data scale."""
    h, w = img_bgr.shape[:2]
    pad = int(radius * 1.35)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)
    crop = img_bgr[y1:y2, x1:x2]
    return crop


def classify_regions(model, img_bgr, circles):
    crops = [prepare_crop(img_bgr, *circle) for circle in circles]
    for start in range(0, len(crops), BATCH_SIZE):
        batch = crops[start:start + BATCH_SIZE]
        results = model.predict(batch, imgsz=CROP_SIZE, conf=CONF_THRESH, verbose=False)
        for crop, result in zip(batch, results):
            if result.boxes is not None and len(result.boxes) > 0:
                best_idx = int(result.boxes.conf.argmax())
                cls_id = int(result.boxes.cls[best_idx])
                conf = float(result.boxes.conf[best_idx])
                label = result.names[cls_id] if cls_id < len(result.names) else "undetected"
                yield label, conf
            else:
                yield "undetected", 0.0


def detect_and_classify(img_bgr, debug=False):
    t0 = time.perf_counter()
    circles = detect_nozzle_regions(img_bgr, debug=debug)
    labels = classify_regions(get_model(), img_bgr, circles) if circles else ()
    detections = []
    class_counts = {c: 0 for c in CLASSES}
    annotated = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    for (cx, cy, radius), (label, conf) in zip(circles, labels):
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w - 1, cx + radius), min(h - 1, cy + radius)
        color = COLORS.get(label, (128, 128, 128))
        needs_review = conf < REVIEW_BAND

        # Draw box + label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        flag = " ?" if needs_review else ""
        tag = f"{label} {conf:.2f}" if conf > 0 else "?"
        tag += " ?" if needs_review else ""
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_x = max(0, min(x1, w - tw - 4))
        tag_y = max(y1, th + 6)
        cv2.rectangle(annotated, (tag_x, tag_y - th - 6), (tag_x + tw + 4, tag_y), color, -1)
        cv2.putText(annotated, tag, (tag_x + 2, tag_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "label": label, "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2], "needs_review": needs_review
        })
        class_counts[label] = class_counts.get(label, 0) + 1

    latency = (time.perf_counter() - t0) * 1000

    # HUD
    nozzle_total = sum(v for k, v in class_counts.items() if k != "unknown")
    hud = [
        f"Identified: {nozzle_total}   Review: {sum(d['needs_review'] for d in detections)}   {latency:.0f} ms",
    ] + [f"  {k}: {v}" for k, v in class_counts.items() if v > 0]

    for i, line in enumerate(hud):
        y = 25 + i * 22
        cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated, detections, class_counts, latency


# ── IMAGE MODE ────────────────────────────────────────────────────────────────

def predict(image_path: str, show=True, debug=False) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    annotated, detections, class_counts, latency = detect_and_classify(img, debug=debug)

    out_path = OUTPUT_DIR / f"{Path(image_path).stem}_detected.jpg"
    if not cv2.imwrite(str(out_path), annotated):
        raise OSError(f"Cannot save: {out_path}")
    if show:
        try: os.startfile(str(out_path))
        except: pass

    return {
        "total_detected": len(detections),
        "class_counts"  : {k: v for k, v in class_counts.items() if v > 0},
        "needs_review"  : any(d["needs_review"] for d in detections),
        "latency_ms"    : round(latency, 1),
        "annotated_path": str(out_path),
        "detections"    : detections,
    }


# ── WEBCAM MODE ───────────────────────────────────────────────────────────────

def run_webcam():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_ID}"); return

    print("\n  Live Detection | Q: Quit  S: Save  P: Pause\n")
    paused = False; snap = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: continue
            annotated, _, _, _ = detect_and_classify(frame)
            cv2.putText(annotated, "Q:Quit S:Save P:Pause",
                        (8, annotated.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("Lechler Nozzle Detection", annotated)
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('s'):
            p = OUTPUT_DIR / f"snap_{snap:04d}.jpg"
            cv2.imwrite(str(p), annotated); snap += 1; print(f"  Saved → {p}")
        elif key == ord('p'):
            paused = not paused; print(f"  {'PAUSED' if paused else 'RESUMED'}")

    cap.release(); cv2.destroyAllWindows()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--webcam" in sys.argv:
        run_webcam()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        if args:
            img_path = args[0]
        else:
            candidates = sorted((PROJECT_ROOT / "data").glob("WIN_*.jpg"))
            if not candidates:
                print("Usage: python main.py <image.jpg>  |  --webcam  |  --debug")
                sys.exit(1)
            img_path = str(candidates[0])
            print(f"[INFO] Using: {img_path}\n")

        result = predict(img_path, show="--no-show" not in sys.argv, debug=DEBUG)

        print(f"\n{'='*55}")
        print(f"  NOZZLE DETECTION RESULTS")
        print(f"{'='*55}")
        print(f"  Total Detected : {result['total_detected']}")
        print(f"  Latency        : {result['latency_ms']} ms")
        print(f"  Needs Review?  : {'YES' if result['needs_review'] else 'NO'}")
        print(f"\n  Counts by Type:")
        for k, v in result["class_counts"].items():
            print(f"    {k:12s}: {v}  {'█' * v}")
        print(f"\n  Saved → {result['annotated_path']}")
        print(f"{'='*55}\n")
