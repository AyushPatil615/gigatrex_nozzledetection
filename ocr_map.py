"""
OCR-Based Nozzle Mapping
========================
Stage 1 : Hough Circles → find occupied tray holes
Stage 2a: EasyOCR       → try to read part-number text on the nozzle
Stage 2b: YOLOv8s       → visual classification fallback when OCR finds nothing

Lechler nozzle codes stamped on parts (expand this dict as you discover more):
  460.668.17  → nozzle_1   (smooth silver ball)
  460.668.19  → nozzle_1
  460.600.17  → nozzle_2   (perforated dome)
  460.600.30  → nozzle_2
  460.608.AA  → nozzle_3   (gold flat-top with slots)
  460.624.AA  → nozzle_4   (gold spiral)
  460.656.AA  → nozzle_5   (silver flat disc with centre hole)

Usage:
    python ocr_map.py                        # auto-picks first WIN_*.jpg in data/
    python ocr_map.py data/my_tray.jpg
    python ocr_map.py --debug                # saves Stage-1 circle debug image
    python ocr_map.py --ocr-only             # skip YOLO fallback, show raw OCR hits
"""

import cv2
import numpy as np
import re
import time
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
_v2 = PROJECT_ROOT / "runs" / "nozzle_detect_v2" / "weights" / "best.pt"
_v1 = PROJECT_ROOT / "runs" / "nozzle_detect_v1" / "weights" / "best.pt"
MODEL_PATH   = _v2 if _v2.exists() else _v1   # auto-pick v2 when training finishes
OUTPUT_DIR   = PROJECT_ROOT / "detect"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES     = ["nozzle_1", "nozzle_2", "nozzle_3", "nozzle_4", "nozzle_5"]
CONF_THRESH = 0.45          # YOLO confidence threshold (fallback)
REVIEW_BAND = 0.65

# ── PART-NUMBER → CLASS MAP ────────────────────────────────────────────────────
# Add every code you spot on the actual nozzles here.
# Keys are lowercased, dots and spaces stripped → "46066817" style.
# Partial matches also work (e.g. "668" alone maps to nozzle_1).
PART_NUMBER_MAP = {
    # nozzle_1 — smooth silver ball
    "46066817": "nozzle_1",
    "46066819": "nozzle_1",
    "46066820": "nozzle_1",
    "46066830": "nozzle_1",
    "668":      "nozzle_1",

    # nozzle_2 — perforated dome
    "46060017": "nozzle_2",
    "46060030": "nozzle_2",
    "460600":   "nozzle_2",
    "600":      "nozzle_2",

    # nozzle_3 — gold flat-top with slots
    "460608":   "nozzle_3",
    "608":      "nozzle_3",

    # nozzle_4 — gold spiral
    "460624":   "nozzle_4",
    "624":      "nozzle_4",

    # nozzle_5 — silver flat disc with centre hole
    "460656":   "nozzle_5",
    "656":      "nozzle_5",
}

COLORS = {
    "nozzle_1": (180, 180, 180),
    "nozzle_2": (80,  200,  80),
    "nozzle_3": (255, 220, 100),
    "nozzle_4": (0,   190, 220),
    "nozzle_5": (200,  80, 200),
    "unknown" : (0,     0, 255),
    "ocr_hit" : (0,   220, 255),   # confirmed by OCR → cyan border
}

DEBUG    = "--debug"    in sys.argv
OCR_ONLY = "--ocr-only" in sys.argv

_ocr_reader = None
_yolo_model = None


# ── HELPERS ────────────────────────────────────────────────────────────────────

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # English only; gpu=True if CUDA is available
        import torch
        gpu = torch.cuda.is_available()
        print(f"[OCR] Loading EasyOCR (gpu={gpu}) …")
        _ocr_reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        print("[OCR] Ready.")
    return _ocr_reader


def get_model():
    global _yolo_model
    if _yolo_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun: python train.py")
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model


def normalize_code(raw: str) -> str:
    """Strip spaces, dots, dashes → lowercase digits only."""
    return re.sub(r"[^0-9a-z]", "", raw.lower())


def lookup_part(raw_text: str):
    """Try to match raw OCR text to a known Lechler part number.
    Returns (class_label, matched_key) or (None, None)."""
    code = normalize_code(raw_text)
    # Exact match first
    if code in PART_NUMBER_MAP:
        return PART_NUMBER_MAP[code], code
    # Partial match: check if any dict key is a substring of the OCR text
    for key, label in PART_NUMBER_MAP.items():
        if key in code:
            return label, key
    return None, None


# ── STAGE 1: HOUGH CIRCLE DETECTION ───────────────────────────────────────────

def detect_nozzle_regions(img_bgr, debug=False):
    h, w = img_bgr.shape[:2]
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    min_r    = int(w * 0.025)
    max_r    = int(w * 0.065)
    min_dist = int(w * 0.07)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_dist,
        param1=60, param2=35,
        minRadius=min_r, maxRadius=max_r,
    )

    boxes     = []
    debug_img = img_bgr.copy() if debug else None

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower   = np.array([90,  40,  10])
        upper   = np.array([140, 255, 140])

        for (cx, cy, r) in circles:
            pad  = int(r * 0.55)
            x1   = max(0, cx - pad);  y1 = max(0, cy - pad)
            x2   = min(w, cx + pad);  y2 = min(h, cy + pad)
            crop = hsv[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            mask       = cv2.inRange(crop, lower, upper)
            hole_ratio = np.count_nonzero(mask) / (crop.shape[0] * crop.shape[1] + 1e-5)
            is_empty   = hole_ratio > 0.60

            if debug_img is not None:
                col = (0, 0, 200) if is_empty else (0, 200, 0)
                cv2.circle(debug_img, (cx, cy), r, col, 2)
                cv2.putText(debug_img, f"{hole_ratio:.2f}", (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

            if not is_empty:
                pad2 = int(r * 1.15)
                boxes.append((max(0, cx - pad2), max(0, cy - pad2),
                              min(w, cx + pad2), min(h, cy + pad2)))

    if debug_img is not None:
        out = OUTPUT_DIR / "debug_circles.jpg"
        cv2.imwrite(str(out), debug_img)
        try: os.startfile(str(out))
        except: pass
        print(f"[DEBUG] Circles saved → {out}")
        print(f"[DEBUG] Total circles : {0 if circles is None else len(circles)}")
        print(f"[DEBUG] Nozzle crops  : {len(boxes)}")

    return boxes


# ── STAGE 2a: OCR on crop ──────────────────────────────────────────────────────

def preprocess_for_ocr(crop_bgr):
    """Enhance contrast and binarise for better OCR on metallic surfaces."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE to lift engraved text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    # Upscale — EasyOCR prefers ≥32 px text height
    scale = max(1, 128 // max(eq.shape))
    if scale > 1:
        eq = cv2.resize(eq, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Otsu binarise on inverted (engraved text = dark on metal)
    _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def ocr_crop(crop_bgr):
    """
    Returns (label, raw_text, matched_key) if a known part number is found,
    else (None, raw_text, None).
    """
    reader  = get_ocr()
    prep    = preprocess_for_ocr(crop_bgr)
    results = reader.readtext(prep, detail=1, paragraph=False,
                              allowlist="0123456789abcdefghijklmnopqrstuvwxyz. -")
    raw_parts = [txt for (_, txt, conf) in results if conf > 0.25]
    raw_text  = " ".join(raw_parts).strip()

    if not raw_text:
        return None, "", None

    label, key = lookup_part(raw_text)
    return label, raw_text, key


# ── STAGE 2b: YOLO FALLBACK ────────────────────────────────────────────────────

def yolo_classify(crop_bgr):
    model  = get_model()
    result = model(crop_bgr, verbose=False, conf=0.10)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        confs  = result.boxes.conf
        best   = int(confs.argmax())
        cls_id = int(result.boxes.cls[best])
        conf   = float(result.boxes.conf[best])
        label  = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
        return label, conf
    return "unknown", 0.0


# ── COMBINED PIPELINE ──────────────────────────────────────────────────────────

def detect_and_map(img_bgr, debug=False):
    t0        = time.time()
    boxes     = detect_nozzle_regions(img_bgr, debug=debug)
    annotated = img_bgr.copy()
    detections   = []
    class_counts = {c: 0 for c in CLASSES}
    ocr_hits     = 0

    for (x1, y1, x2, y2) in boxes:
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        source = "yolo"
        conf   = 0.0

        # ── Try OCR first ──────────────────────────────────────────────────
        ocr_label, raw_text, matched_key = ocr_crop(crop)

        if ocr_label is not None:
            label  = ocr_label
            conf   = 1.0          # OCR match = certain
            source = "ocr"
            ocr_hits += 1
            print(f"  [OCR ✓] '{raw_text}' → matched '{matched_key}' → {label}")
        elif raw_text:
            # OCR found text but no known code — still show it
            label  = "unknown"
            conf   = 0.5
            source = "ocr_unknown"
            print(f"  [OCR ?] '{raw_text}' — no match in PART_NUMBER_MAP")
        else:
            # No text found — fall back to YOLO
            if not OCR_ONLY:
                label, conf = yolo_classify(crop)
                if conf < CONF_THRESH:
                    label = "unknown"
                # Suppress low-conf unknowns (empty holes)
                if label == "unknown" and conf < 0.70:
                    continue
            else:
                continue

        # ── Determine display colour ───────────────────────────────────────
        color        = COLORS.get(label, (128, 128, 128))
        needs_review = (source != "ocr") and (conf < REVIEW_BAND)
        border_w     = 3 if source == "ocr" else 2

        # OCR hits get a cyan outer ring to visually distinguish them
        if source == "ocr":
            cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2),
                          COLORS["ocr_hit"], 1)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, border_w)
        flag = " ?" if needs_review else ""
        src_tag = "OCR" if source == "ocr" else "AI"
        tag  = f"[{src_tag}] {label} {conf:.2f}{flag}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, tag, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        detections.append({
            "label": label, "confidence": round(conf, 3),
            "source": source, "ocr_text": raw_text,
            "bbox": [x1, y1, x2, y2], "needs_review": needs_review,
        })
        class_counts[label] = class_counts.get(label, 0) + 1

    latency = (time.time() - t0) * 1000

    # ── HUD ───────────────────────────────────────────────────────────────
    nozzle_total = sum(v for k, v in class_counts.items() if k != "unknown")
    hud_lines = [
        f"Nozzles: {nozzle_total}  Unknown: {class_counts.get('unknown',0)}"
        f"  OCR hits: {ocr_hits}  Latency: {latency:.0f}ms",
    ] + [f"  {k}: {v}" for k, v in class_counts.items() if v > 0]

    for i, line in enumerate(hud_lines):
        y = 25 + i * 22
        cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated, detections, class_counts, latency


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        img_path = args[0]
    else:
        candidates = sorted((PROJECT_ROOT / "data").glob("WIN_*.jpg"))
        if not candidates:
            print("Usage: python ocr_map.py <image.jpg>  [--debug] [--ocr-only]")
            sys.exit(1)
        img_path = str(candidates[0])
        print(f"[INFO] Using: {img_path}\n")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read: {img_path}"); sys.exit(1)

    annotated, detections, class_counts, latency = detect_and_map(img, debug=DEBUG)

    out_path = OUTPUT_DIR / f"{Path(img_path).stem}_ocr.jpg"
    cv2.imwrite(str(out_path), annotated)
    try: os.startfile(str(out_path))
    except: pass

    print(f"\n{'='*55}")
    print(f"  OCR + AI NOZZLE MAPPING RESULTS")
    print(f"{'='*55}")
    print(f"  Total Detected : {len(detections)}")
    print(f"  Latency        : {latency:.1f} ms")
    print(f"\n  Counts by Type:")
    for k, v in class_counts.items():
        if v > 0:
            print(f"    {k:12s}: {v}  {'█' * v}")
    print(f"\n  OCR hits  : {sum(1 for d in detections if d['source']=='ocr')}")
    print(f"  AI  hits  : {sum(1 for d in detections if d['source']=='yolo')}")
    print(f"\n  Detections:")
    for d in detections:
        ocr_note = f"  (OCR: '{d['ocr_text']}')" if d["ocr_text"] else ""
        print(f"    [{d['source'].upper():12s}] {d['label']:10s} conf={d['confidence']:.2f}{ocr_note}")
    print(f"\n  Saved → {out_path}")
    print(f"{'='*55}\n")
    print("TIP: Add codes from above OCR hits to PART_NUMBER_MAP in ocr_map.py")
