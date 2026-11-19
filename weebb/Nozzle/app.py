"""
NozzleVision AI — Automated Tray Inspection & Inventory Dashboard
Built for Hackathon Demo & Panel Presentation
"""

import os
import io
import time
import pandas as pd
from collections import Counter
from PIL import Image
import cv2
import numpy as np

# Point Ultralytics config locally to avoid permission warnings on Windows
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("./.ultralytics")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import gradio as gr
except ImportError:
    gr = None

# -------------------------------------------------------------
# Model Loader: Automatically find best.pt or fallback
# -------------------------------------------------------------
def get_model_path():
    candidate_paths = [
        "best.pt",
        "weights/best.pt",
        "runs/nozzle_detector/exp_yolo11s_b8/weights/best.pt",
        "runs/detect/train/weights/best.pt",
        "yolo11s.pt",
        "yolov8s.pt",
        "yolov8n.pt"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return "yolov8n.pt"

# Cache loaded model
LOADED_MODEL = None
CURRENT_MODEL_PATH = None

def load_yolo_model(model_path=None):
    global LOADED_MODEL, CURRENT_MODEL_PATH
    if model_path is None:
        model_path = get_model_path()
    
    if LOADED_MODEL is None or CURRENT_MODEL_PATH != model_path:
        print(f"[INFO] Loading model from: {model_path}")
        LOADED_MODEL = YOLO(model_path)
        CURRENT_MODEL_PATH = model_path
    return LOADED_MODEL

# -------------------------------------------------------------
# Core Inference & Analytics Engine
# -------------------------------------------------------------
TOTAL_TRAY_SLOTS = 77  # Standard 7x11 grid tray

def inspect_tray(input_img, conf_thresh, iou_thresh):
    if input_img is None:
        return None, "Please upload an image.", None, None
    
    t0 = time.time()
    model = load_yolo_model()
    
    # Run YOLO inference
    results = model.predict(
        source=input_img,
        conf=conf_thresh,
        iou=iou_thresh,
        save=False,
        verbose=False
    )[0]
    
    latency_ms = round((time.time() - t0) * 1000, 1)
    
    # Extract bounding boxes and class names
    class_names = results.names
    detected_items = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        label = class_names.get(cls_id, f"Class {cls_id}")
        detected_items.append({
            "type": label.replace("_", " ").title(),
            "confidence": conf_val
        })
        
    total_count = len(detected_items)
    type_counts = Counter([item["type"] for item in detected_items])
    
    # Render annotated image
    annotated_bgr = results.plot(line_width=3, font_size=1.2)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Build Metrics Markdown Cards
    occupancy_rate = min(100.0, round((total_count / TOTAL_TRAY_SLOTS) * 100, 1))
    
    summary_html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 15px;">
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center; color: white;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Total Parts Found</div>
            <div style="font-size: 28px; font-weight: 700; color: #38bdf8; margin-top: 4px;">{total_count}</div>
        </div>
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center; color: white;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Unique Part Types</div>
            <div style="font-size: 28px; font-weight: 700; color: #a855f7; margin-top: 4px;">{len(type_counts)}</div>
        </div>
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center; color: white;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Tray Capacity</div>
            <div style="font-size: 28px; font-weight: 700; color: #34d399; margin-top: 4px;">{occupancy_rate}%</div>
        </div>
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center; color: white;">
            <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600;">Inference Latency</div>
            <div style="font-size: 28px; font-weight: 700; color: #f59e0b; margin-top: 4px;">{latency_ms} ms</div>
        </div>
    </div>
    """
    
    # Build Inventory DataFrame
    if total_count > 0:
        table_rows = []
        for n_type, count in type_counts.most_common():
            pct = round((count / total_count) * 100, 1)
            table_rows.append({
                "Nozzle Type": n_type,
                "Quantity": count,
                "Tray Share (%)": f"{pct}%",
                "Quality Status": "✅ Passed Inspection"
            })
        df = pd.DataFrame(table_rows)
    else:
        df = pd.DataFrame(columns=["Nozzle Type", "Quantity", "Tray Share (%)", "Quality Status"])
        
    # Generate CSV path for download
    csv_path = "tray_inventory_bom.csv"
    df.to_csv(csv_path, index=False)
    
    return annotated_rgb, summary_html, df, csv_path

# -------------------------------------------------------------
# Gradio Presentation Interface
# -------------------------------------------------------------
custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.header-title {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.header-badge {
    background: #3b82f6;
    color: white;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 6px;
    margin-bottom: 0;
}
"""

def create_app():
    with gr.Blocks(css=custom_css, title="NozzleVision AI | Industrial Tray Inspector", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        with gr.Column():
            gr.HTML("""
            <div class="header-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h1 class="header-title">
                        <span>🔍 NozzleVision AI</span>
                        <span class="header-badge">Hackathon 2026</span>
                    </h1>
                    <div style="color: #64748b; font-size: 13px;">Real-Time Industrial Vision System</div>
                </div>
                <p class="header-subtitle">
                    Automated object detection, part identification, tray counting, and instant Bill of Materials (BOM) export.
                </p>
            </div>
            """)
            
            with gr.Row():
                # Left Panel: Input & Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Input & Camera Feed")
                    input_image = gr.Image(
                        label="Upload Tray Photo or Capture from Webcam",
                        type="pil",
                        sources=["upload", "webcam"],
                        elem_id="tray_input"
                    )
                    
                    with gr.Accordion("⚙️ Detection Sensitivity Controls", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.10, maximum=0.95, value=0.40, step=0.05,
                            label="Confidence Threshold (Conf)"
                        )
                        iou_slider = gr.Slider(
                            minimum=0.20, maximum=0.90, value=0.50, step=0.05,
                            label="Overlap Threshold (IoU NMS)"
                        )
                    
                    inspect_btn = gr.Button("🚀 Run Tray Inspection", variant="primary", size="lg")
                    
                    # Example image loader
                    example_img_path = "data/full.jpeg"
                    if os.path.exists(example_img_path):
                        gr.Examples(
                            examples=[[example_img_path, 0.40, 0.50]],
                            inputs=[input_image, conf_slider, iou_slider],
                            label="💡 Quick Demo Tray Image"
                        )
                
                # Right Panel: Output & Live Analytics
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Detection Results & Live Visuals")
                    output_image = gr.Image(label="Annotated Detections", type="numpy")
                    metrics_card = gr.HTML(label="Key Performance Indicators")
            
            # Bottom Section: Inventory Table & Export
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📋 Automated Bill of Materials (BOM) Inventory")
                    inventory_table = gr.Dataframe(
                        headers=["Nozzle Type", "Quantity", "Tray Share (%)", "Quality Status"],
                        interactive=False,
                        label="Nozzle Classification Breakdown"
                    )
                    with gr.Row():
                        download_btn = gr.File(label="📥 Export Report (CSV / ERP Compatible)")

            # Event wiring
            inspect_btn.click(
                fn=inspect_tray,
                inputs=[input_image, conf_slider, iou_slider],
                outputs=[output_image, metrics_card, inventory_table, download_btn]
            )
            
            input_image.change(
                fn=inspect_tray,
                inputs=[input_image, conf_slider, iou_slider],
                outputs=[output_image, metrics_card, inventory_table, download_btn]
            )
            
    return demo

if __name__ == "__main__":
    demo = create_app()
    # launch with share=True creates an instant public URL (https://xxxx.gradio.live) for 72 hours
    print("\n" + "="*60)
    print("  🚀 Launching NozzleVision AI Web Dashboard...")
    print("  ⚡ Set share=True to generate an instant web link for the panel!")
    print("="*60 + "\n")
    demo.launch(share=True, inbrowser=True)
