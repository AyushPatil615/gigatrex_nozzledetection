# 🔬 NozzleVision AI — Lechler Nozzle Detection System

> **Gigatrex Hackathon Submission** — Automated industrial tray inspection using YOLOv8s + Hough Circle detection, with a React web app and live webcam support.

---

## 📸 Demo

| File Upload Detection | Live Webcam |
|---|---|
| Upload a tray photo → get annotated image with nozzle counts | External webcam streams to browser at `localhost:5050` |

---

## 🏗️ Architecture

```
lechler_detect/
├── main.py                        # Core pipeline: Hough Circles → YOLO classify
├── train.py                       # YOLOv8s training script
├── webcam_detect.py               # Live webcam → browser stream (no cv2.imshow)
├── ocr_map.py                     # OCR + YOLO hybrid (supplementary)
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── yolo_dataset/              # YOLO-format dataset (288 train / 53 val)
│   │   ├── data.yaml              # nc=5, names=[nozzle_1..nozzle_5]
│   │   ├── images/train|val/
│   │   └── labels/train|val/
│   ├── images/                    # Original nozzle reference images
│   └── annotated/                 # Annotated training images
│
├── data_pipeline/                 # Dataset preparation scripts
│   ├── augment_dataset.py
│   ├── auto_annotate.py
│   ├── prepare_yolo_split.py
│   └── ...
│
├── runs/
│   ├── nozzle_detect_v1/weights/best.pt   # v1 model (6-class, includes unknown)
│   └── nozzle_detect_v2/weights/best.pt   # v2 model (5-class, production) ✅
│
└── weebb/Nozzle/
    ├── backend/
    │   └── server.py              # FastAPI backend (calls root main.py)
    └── frontend/                  # React + Vite + Tailwind web app
        ├── src/
        │   ├── App.tsx            # Main app with file upload + live camera
        │   └── components/
        │       └── LiveCamera.tsx # Webcam device selector + stream
        ├── vercel.json            # Vercel deployment config
        └── package.json
```

---

## 🧠 Detection Pipeline

```
Input Image (tray photo)
        │
        ▼
Stage 1: Hough Circle Transform
  → Finds all nozzle holes in the tray
  → Filters empty holes by darkness ratio (EMPTY_RATIO=0.48)
        │
        ▼
Stage 2: YOLOv8s Classification
  → Crops each nozzle region (pad=1.35×radius, 640px)
  → Classifies into one of 5 types
  → Confidence threshold: 0.20
        │
        ▼
Output: Annotated image + class counts + latency
```

**5 Nozzle Classes:**
| Class | Description |
|-------|-------------|
| `nozzle_1` | Smooth Dome |
| `nozzle_2` | Perforated Dome |
| `nozzle_3` | Gold Flat-Top |
| `nozzle_4` | Gold Spiral |
| `nozzle_5` | Silver Disc |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics fastapi uvicorn opencv-python numpy pillow pytesseract
```

### 2. Train the Model (optional — pre-trained weights included)

```bash
python train.py
# Outputs: runs/nozzle_detect_v2/weights/best.pt
```

### 3. Run Detection on an Image

```bash
python main.py data/WIN_20261119_11_27_19_Pro.jpg
```

### 4. Live Webcam Detection (browser-based)

```bash
python webcam_detect.py              # auto-detect camera
python webcam_detect.py --camera 1   # explicit camera id (use 1 for external USB webcam)
```
Then open **http://localhost:5050** in your browser.

---

## 🌐 Web App Deployment

### Backend (FastAPI + ngrok)

```bash
# Terminal 1 — Start backend
cd weebb/Nozzle
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000

# Terminal 2 — Expose via ngrok
ngrok config add-authtoken YOUR_NGROK_TOKEN
ngrok http 8000
```

Backend endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status + class list |
| `/api/detect` | POST | File upload → annotated image |
| `/api/detect-frame` | POST | Base64 frame → annotated image |

### Frontend (Vercel)

```bash
cd weebb/Nozzle/frontend
npm install
npm run build

# Deploy to Vercel
vercel login
vercel --prod
```

After deploy:
1. Open your Vercel URL
2. Click the **"Model: Connect Later"** pill (top right)
3. Paste your ngrok URL → **Save & Connect**
4. Upload a tray photo → see real YOLO results ✅

---

## 📊 Model Performance

| Metric | v1 (6-class) | v2 (5-class) ✅ |
|--------|-------------|----------------|
| Training samples | 288 | 288 |
| Validation samples | 53 | 53 |
| Epochs | 50 | 50 |
| Batch size | 8 | 8 |
| GPU | RTX 2050 | RTX 2050 |
| Classes | 6 (incl. unknown) | 5 (clean) |
| Avg detected/tray | ~7 | ~27 |

> **Key fix:** Switched from circular-masked 224px crops → plain rectangular 640px crops. Detection jumped from 7 → 27 nozzles per tray.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8s (Ultralytics) |
| Circle Detection | OpenCV Hough Circle Transform |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 19 + Vite + Tailwind CSS |
| Tunnel | ngrok |
| Frontend Deploy | Vercel |
| OCR (supplementary) | Tesseract via pytesseract |
| GPU | CUDA 12.4, PyTorch 2.6.0+cu124 |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | Core Hough + YOLO pipeline |
| [`train.py`](train.py) | Model training |
| [`webcam_detect.py`](webcam_detect.py) | Live webcam → browser stream |
| [`ocr_map.py`](ocr_map.py) | OCR-based part number identification |
| [`weebb/Nozzle/backend/server.py`](weebb/Nozzle/backend/server.py) | FastAPI server |
| [`weebb/Nozzle/frontend/src/App.tsx`](weebb/Nozzle/frontend/src/App.tsx) | React web app |
| [`data/yolo_dataset/data.yaml`](data/yolo_dataset/data.yaml) | Dataset config |
| [`runs/nozzle_detect_v2/weights/best.pt`](runs/nozzle_detect_v2/weights/best.pt) | Trained model weights |

---

## 👥 Team — Gigatrex

Built for the **Lechler Image Processing Hackathon**.

---

## 📄 License

MIT
