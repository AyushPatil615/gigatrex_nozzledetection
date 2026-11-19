"""
YOLOv8s Detection Training
============================
Trains YOLOv8s to detect and classify 5 nozzle types simultaneously.
Works for both still images and live webcam inference.

Run:
  python train.py
"""

from ultralytics import YOLO
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML    = PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"
MODEL        = "yolov8s.pt"          # YOLOv8 small — fast + accurate
EPOCHS       = 50
IMG_SIZE     = 640
BATCH        = 8                     # reduce to 4 if GPU OOM
DEVICE       = 0 if torch.cuda.is_available() else "cpu"
PROJECT_DIR  = PROJECT_ROOT / "runs"
RUN_NAME     = "nozzle_detect_v2"


def train():
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "None (CPU only)"
    
    print(f"\n{'='*60}")
    print(f"  YOLOv8s Nozzle Detection Training")
    print(f"  CUDA available: {cuda_ok}")
    print(f"  Device        : {device_name} ({DEVICE})")
    print(f"  Data          : {DATA_YAML}")
    print(f"  Epochs        : {EPOCHS}  |  Batch: {BATCH}  |  ImgSz: {IMG_SIZE}")
    print(f"{'='*60}")

    if not cuda_ok:
        print("\n  [NOTICE] PyTorch is running on CPU because the CUDA-enabled")
        print("  build of PyTorch is not installed in this environment.")
        print("  To enable your GPU (NVIDIA RTX 2050), run in your terminal:")
        print("  pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124\n")

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        print("  → Run: python data_pipeline/prepare_yolo_split.py")
        return

    model = YOLO(MODEL)

    model.train(
        data     = str(DATA_YAML),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        device   = DEVICE,
        patience = 10,
        project  = str(PROJECT_DIR),
        name     = RUN_NAME,
        exist_ok = True,
        # Augmentation settings
        hsv_h    = 0.015,
        hsv_s    = 0.5,
        hsv_v    = 0.4,
        degrees  = 15,
        fliplr   = 0.5,
        scale    = 0.5,
        mosaic   = 0.5,
    )

    best = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\n  Training complete!")
    print(f"  Best weights → {best}")
    print(f"\n  Run webcam:  python webcam_detect.py")
    print(f"  Run on image: python main.py <image_path>\n")


if __name__ == "__main__":
    train()
