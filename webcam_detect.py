"""
Live Webcam Nozzle Detection  —  NozzleVision AI
==================================================
Uses the SAME main.py pipeline as the web backend.
Streams annotated frames to a local HTTP server so you
can view in any browser (no cv2.imshow needed).

Controls (in terminal):
  Ctrl-C  → quit
  s       → save snapshot

Open http://localhost:5050 in your browser to watch.

Run:
  python webcam_detect.py              # tries camera 0, then 1
  python webcam_detect.py --camera 1  # explicit camera id
"""

import sys, io, time, argparse, threading, base64
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-pick v2 model, fall back to v1
_v2 = PROJECT_ROOT / "runs" / "nozzle_detect_v2" / "weights" / "best.pt"
_v1 = PROJECT_ROOT / "runs" / "nozzle_detect_v1" / "weights" / "best.pt"
MODEL_PATH = _v2 if _v2.exists() else _v1

CLASSES = ["nozzle_1", "nozzle_2", "nozzle_3", "nozzle_4", "nozzle_5"]

OUTPUT_DIR = PROJECT_ROOT / "detect"
OUTPUT_DIR.mkdir(exist_ok=True)

PORT = 5050   # browser viewer port

# ── Import main.py pipeline ──────────────────────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_PATH}")
try:
    import main as nozzle_main
    USE_FULL_PIPELINE = hasattr(nozzle_main, "detect_and_classify")
    if USE_FULL_PIPELINE:
        print("[INFO] Using full main.py pipeline (Hough + YOLO)")
    else:
        print("[WARN] detect_and_classify not found — raw YOLO fallback")
except Exception as e:
    print(f"[WARN] Could not import main.py: {e}")
    USE_FULL_PIPELINE = False

if not USE_FULL_PIPELINE:
    from ultralytics import YOLO
    _model = YOLO(str(MODEL_PATH))
    print("[INFO] Raw YOLO fallback loaded")


# ── Shared state between camera thread and HTTP server ───────────────────────
_state = {
    "jpeg_bytes": b"",   # latest annotated frame as JPEG
    "counts": {},
    "latency_ms": 0.0,
    "total": 0,
    "lock": threading.Lock(),
    "snap_count": 0,
    "save_next": False,
}


# ── HTTP server: serves a live-refresh page + MJPEG stream ───────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NozzleVision — Live Detection</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; margin:0; display:flex;
           flex-direction:column; align-items:center; }
    h1   { margin:12px 0 4px; font-size:1.3rem; }
    img  { max-width:95vw; border:2px solid #0ff; border-radius:8px; margin:8px; }
    #info{ font-size:1rem; padding:6px 20px; background:#222; border-radius:6px; margin:4px; }
    button { margin:8px; padding:8px 20px; font-size:1rem; cursor:pointer;
             background:#0aa; border:none; border-radius:6px; color:#000; }
  </style>
</head>
<body>
  <h1>🔬 NozzleVision AI — Live Webcam</h1>
  <img src="/stream" id="feed" alt="stream">
  <div id="info">Connecting…</div>
  <button onclick="fetch('/snap')">📸 Save Snapshot</button>
  <script>
    setInterval(() => {
      fetch('/stats').then(r=>r.json()).then(d=>{
        document.getElementById('info').textContent =
          `Latency: ${d.latency_ms.toFixed(0)} ms  |  ` +
          `Nozzles: ${d.total}  |  ` +
          Object.entries(d.counts).filter(([k,v])=>v>0).map(([k,v])=>`${k}:${v}`).join('  ');
      });
    }, 500);
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # silence HTTP logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _state["lock"]:
                        data = _state["jpeg_bytes"]
                    if data:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                    time.sleep(0.04)   # ~25 fps max
            except Exception:
                pass

        elif self.path == "/stats":
            import json
            with _state["lock"]:
                payload = {
                    "counts": _state["counts"],
                    "latency_ms": _state["latency_ms"],
                    "total": _state["total"],
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        elif self.path == "/snap":
            with _state["lock"]:
                _state["save_next"] = True
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Snapshot queued")

        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    srv.serve_forever()


# ── Camera thread ─────────────────────────────────────────────────────────────
def run_camera(camera_id: int):
    cap = None
    for cid in ([camera_id] if camera_id >= 0 else [0, 1, 2]):
        cap = cv2.VideoCapture(cid)
        if cap.isOpened():
            print(f"[INFO] Camera opened: id={cid}")
            break
        cap.release()
    if not cap or not cap.isOpened():
        print("[ERROR] No camera found. Tried ids 0,1,2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n{'='*55}")
    print(f"  NozzleVision AI  —  Live Webcam")
    print(f"  Model : {MODEL_PATH.name}  ({MODEL_PATH.parent.parent.name})")
    print(f"  View  : http://localhost:{PORT}")
    print(f"  Press Ctrl-C to quit")
    print(f"{'='*55}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying…")
            time.sleep(0.1)
            continue

        t0 = time.time()

        if USE_FULL_PIPELINE:
            # Full Hough + YOLO pipeline — same as web backend
            annotated, detections, class_counts, latency_ms = \
                nozzle_main.detect_and_classify(frame)
        else:
            # Raw YOLO fallback
            results = _model(frame, conf=0.20, verbose=False)
            annotated = results[0].plot()
            class_counts = {}
            for box in results[0].boxes:
                lbl = CLASSES[int(box.cls)] if int(box.cls) < len(CLASSES) else "unknown"
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
            latency_ms = (time.time() - t0) * 1000

        total = sum(v for k, v in class_counts.items() if k not in ("undetected", "unknown"))

        # Overlay latency on frame
        cv2.putText(
            annotated,
            f"Latency: {latency_ms:.0f}ms  |  Nozzles: {total}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            f"Latency: {latency_ms:.0f}ms  |  Nozzles: {total}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA
        )

        # Encode to JPEG
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        with _state["lock"]:
            _state["jpeg_bytes"] = buf.tobytes()
            _state["counts"] = dict(class_counts)
            _state["latency_ms"] = latency_ms
            _state["total"] = total

            if _state["save_next"]:
                snap_path = OUTPUT_DIR / f"snapshot_{_state['snap_count']:04d}.jpg"
                cv2.imwrite(str(snap_path), annotated)
                print(f"  Saved → {snap_path}")
                _state["snap_count"] += 1
                _state["save_next"] = False

    cap.release()


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera device id (default: auto-detect 0→1→2)")
    args = parser.parse_args()

    # Start HTTP server in background
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    try:
        run_camera(args.camera)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
