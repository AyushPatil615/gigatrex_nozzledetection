"""
NozzleVision AI - FastAPI Backend
Directly calls your root main.py pipeline without modifying or overriding it.
"""

import os, io, time, base64, sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2, numpy as np

# ── Import root main.py directly ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Standard import of your project's main.py
import main

app = FastAPI(title="NozzleVision AI Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TOTAL_TRAY_SLOTS = 77


# ── Core inference directly using main.py ─────────────────────────────────────
def run_pipeline(img_bgr: np.ndarray):
    """
    Executes the exact detect_and_classify function from main.py
    with main.py's native configuration.
    """
    # Direct call to your main code
    annotated, detections, class_counts, latency_ms = main.detect_and_classify(img_bgr)

    # Encode annotated image -> base64 for frontend
    _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    # Friendly display labels
    display_names = {
        "nozzle_1": "Nozzle 1 (Smooth Dome)",
        "nozzle_2": "Nozzle 2 (Perforated Dome)",
        "nozzle_3": "Nozzle 3 (Gold Flat-Top)",
        "nozzle_4": "Nozzle 4 (Gold Spiral)",
        "nozzle_5": "Nozzle 5 (Silver Disc)",
    }

    # Count identified nozzles (excluding undetected holes)
    total_nozzles = sum(v for k, v in class_counts.items() if k not in ("undetected", "unknown"))
    total_denom = max(1, total_nozzles)

    breakdown = []
    for cls in main.CLASSES:
        count = class_counts.get(cls, 0)
        if count > 0:
            breakdown.append({
                "type": display_names.get(cls, cls),
                "class": cls,
                "count": count,
                "percentage": round((count / total_denom) * 100, 1),
            })

    # Model version check from main.py's MODEL_PATH
    model_name = main.MODEL_PATH.parent.parent.name  # nozzle_detect_v2 or nozzle_detect_v1

    return {
        "success": True,
        "latency_ms": round(latency_ms, 1),
        "total_count": total_nozzles,
        "undetected": class_counts.get("undetected", 0),
        "unique_types": len(breakdown),
        "occupancy_rate": min(100.0, round((total_nozzles / TOTAL_TRAY_SLOTS) * 100, 1)),
        "total_slots": TOTAL_TRAY_SLOTS,
        "model_version": model_name,
        "breakdown": breakdown,
        "detections": detections,
        "annotated_image": annotated_b64,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "NozzleVision AI",
        "model_path": str(main.MODEL_PATH),
        "model_exists": main.MODEL_PATH.exists(),
        "classes": main.CLASSES,
    }


class Base64FrameRequest(BaseModel):
    image_base64: str


@app.post("/api/detect-frame")
async def detect_frame(payload: Base64FrameRequest):
    """Live camera stream endpoint."""
    try:
        raw = payload.image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Invalid image data")
        return JSONResponse(run_pipeline(img))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/detect")
async def detect_file(file: UploadFile = File(...)):
    """File upload endpoint."""
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode uploaded image")
        return JSONResponse(run_pipeline(img))
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
