"""
Lechler Nozzle Detection — Streamlit Web App
=============================================
Run:
    streamlit run app.py

Opens in browser at http://localhost:8501
Upload a tray image → get annotated output + counts table instantly.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from pathlib import Path
import sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lechler Nozzle Detector",
    page_icon="🔩",
    layout="wide",
)

st.title("🔩 Lechler Nozzle Detection System")
st.markdown("Upload a tray image to detect and classify all nozzle types.")

# ── Import backend ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from main import detect_and_classify, CLASSES, COLORS
    backend_ok = True
except Exception as e:
    backend_ok = False
    st.error(f"❌ Backend import failed: {e}")

# ── Sidebar — config ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

conf_thresh = st.sidebar.slider(
    "Confidence Threshold", 0.10, 0.90, 0.30, 0.05,
    help="Only show detections above this confidence level"
)

show_debug = st.sidebar.checkbox("Show Stage-1 Circle Debug", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Class Legend**")
class_colors = {
    "nozzle_1": "#B4B4B4",
    "nozzle_2": "#50C850",
    "nozzle_3": "#FFDC64",
    "nozzle_4": "#00BEDC",
    "nozzle_5": "#C850C8",
}
for name, color in class_colors.items():
    st.sidebar.markdown(
        f'<span style="background:{color};padding:2px 10px;border-radius:4px;'
        f'color:#000;font-weight:bold">{name}</span>',
        unsafe_allow_html=True,
    )

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a tray image here",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG and PNG"
)

# ── Demo images ───────────────────────────────────────────────────────────────
demo_images = sorted(Path("data").glob("WIN_*.jpg")) + \
              sorted(Path("data").glob("WhatsApp*.jpeg"))

if not uploaded and demo_images:
    st.markdown("**Or pick a demo image:**")
    cols = st.columns(min(len(demo_images), 4))
    for i, img_path in enumerate(demo_images[:4]):
        with cols[i]:
            if st.button(img_path.name[:30], key=f"demo_{i}"):
                st.session_state["demo_path"] = str(img_path)

# ── Run detection ─────────────────────────────────────────────────────────────
img_bgr = None

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif "demo_path" in st.session_state:
    img_bgr = cv2.imread(st.session_state["demo_path"])

if img_bgr is not None and backend_ok:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with st.spinner("🔍 Detecting nozzles…"):
        t0 = time.time()

        # Temporarily override CONF_THRESH from slider
        import main as _main
        _orig_thresh = _main.CONF_THRESH
        _main.CONF_THRESH = conf_thresh

        try:
            annotated, detections, class_counts, latency = detect_and_classify(
                img_bgr, debug=show_debug
            )
        finally:
            _main.CONF_THRESH = _orig_thresh

    with col2:
        st.subheader("✅ Detection Result")
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    # ── Metrics row ────────────────────────────────────────────────────────
    st.markdown("---")
    nozzle_total = sum(v for k, v in class_counts.items() if k not in ("undetected",))
    undetected   = sum(v for k, v in class_counts.items() if k == "undetected")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Detected",    nozzle_total)
    m2.metric("Undetected",        undetected)
    m3.metric("Latency",           f"{latency:.0f} ms")
    m4.metric("Needs Review",      sum(1 for d in detections if d["needs_review"]))

    # ── Counts table ───────────────────────────────────────────────────────
    st.markdown("### 📊 Nozzle Counts")
    count_cols = st.columns(5)
    nozzle_classes = [c for c in CLASSES if c not in ("unknown", "undetected")]
    for i, cls in enumerate(nozzle_classes):
        count = class_counts.get(cls, 0)
        color = class_colors.get(cls, "#888")
        count_cols[i].markdown(
            f'<div style="background:{color};padding:12px;border-radius:8px;'
            f'text-align:center;color:#000">'
            f'<b>{cls}</b><br><span style="font-size:2em">{count}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Detection detail table ─────────────────────────────────────────────
    if detections:
        st.markdown("### 🗂️ Detection Details")
        import pandas as pd
        rows = []
        for i, d in enumerate(detections):
            rows.append({
                "#":           i + 1,
                "Label":       d["label"],
                "Confidence":  f"{d['confidence']:.2f}",
                "Review?":     "⚠️ YES" if d["needs_review"] else "✅ OK",
                "BBox":        str(d["bbox"]),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            file_name="nozzle_detections.csv",
            mime="text/csv",
        )

    # ── Save annotated image ───────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "detect"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "web_result.jpg"
    cv2.imwrite(str(out_path), annotated)

    # Download annotated image
    with open(out_path, "rb") as f:
        st.download_button(
            "⬇️ Download Annotated Image",
            f.read(),
            file_name="nozzle_detected.jpg",
            mime="image/jpeg",
        )

elif img_bgr is None:
    st.info("👆 Upload an image or select a demo image to start detection.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
_v2 = PROJECT_ROOT / "runs" / "nozzle_detect_v2" / "weights" / "best.pt"
_v1 = PROJECT_ROOT / "runs" / "nozzle_detect_v1" / "weights" / "best.pt"
model_ver = "v2 ✅" if _v2.exists() else "v1 (v2 training…)"
st.caption(f"Model: {model_ver} | Team Gigatrex | Lechler Hackathon 2026")
