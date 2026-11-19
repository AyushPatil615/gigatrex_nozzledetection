# Lechler AI Hackathon 2026 — Image Processing Playbook (v2)

> **Team gigatrex · 2 people · one day.**
> Confirmed brief: *develop a computer vision solution capable of detecting and identifying objects in images.* Detailed requirements and data arrive at the venue.
>
> **What changed in v2:** this is no longer only a preparation guide. It is a **winning playbook**: judging-oriented strategy, tiered goals, pivot rules, copy-paste code skeletons, a demo script, and a pitch structure built around industrial value.

---

## Part A. How to win (read this first)

### A1. What judges usually reward

Hackathon juries rarely pick the highest-accuracy model. They pick the team that **solved a real problem, proved it works, and told the story clearly**. Expect to be judged on some mix of:

| Criterion | What it means for us | How we score on it |
|---|---|---|
| Problem understanding | We grasp the industrial need, not just the ML task | Open the pitch with the business problem; ask sharp mentor questions at 9:00 |
| Technical soundness | Right method for the data, honest evaluation | Baseline → improvement → measured gain; no leakage; no invented metrics |
| Working demo | It runs live, on unseen images | One-click demo, offline, backup images, tested on 10+ samples |
| Results | Numbers *and* pictures | Metric table + confusion matrix/PR curve + annotated examples |
| Industrial relevance | Could this run in a factory or lab? | Latency, CPU fallback, edge deployment, confidence thresholds, human-in-the-loop |
| Communication | Clear in 5 minutes | Fixed 8-slide structure (Part H), rehearsed twice |
| Creativity / extra mile | Something memorable | One "wow" feature, chosen from Tier 3 below, *only after* Tier 1 and 2 are solid |

**Ask the organizers for the actual scoring rubric at the briefing and re-weight this table accordingly.**

### A2. The five things that separate winners from the rest

1. **A working end-to-end pipeline by midday.** A mediocre model in a finished system beats a great model in a notebook.
2. **Honest, well-chosen metrics.** Pick the metric that matches the business cost (usually recall on defects/objects of interest), and show the threshold trade-off.
3. **Failure analysis shown on stage.** Showing *where it breaks and why* signals engineering maturity. Judges trust teams that know their limits.
4. **Industrial framing.** Translate the output into value: faster inspection, fewer missed defects, less manual effort, traceability.
5. **A calm, rehearsed demo.** The live run must be boring and reliable.

### A3. Lechler context (use as hypotheses, verify at the venue)

Lechler is a spray-nozzle and precision-nozzle manufacturer. Likely data themes: nozzle types and variants, nozzle components, spray patterns, wear/clogging/defects, and product identification. Possible framings to test against the real data:

- **Product identification:** recognise the nozzle type or variant from a photo (support/service use case).
- **Quality inspection:** find defects, burrs, deformation, or contamination on parts.
- **Spray pattern analysis:** classify or measure spray shape, symmetry, or uniformity from images or video.
- **Counting / kitting:** verify the right parts are present on a tray.

Do not force one of these onto the task. Use the vocabulary that fits the real data.

### A4. Tiered goals (decide scope at 10:30, then stick to it)

| Tier | Goal | Done when | Target time |
|---|---|---|---|
| **Tier 1: MVP (must ship)** | Pretrained or simple baseline, image in → structured prediction + annotated image out | `predict()` works on ≥10 unseen images | by 12:00 |
| **Tier 2: Solid** | Fine-tuned / tuned model, proper evaluation, error analysis, simple UI | Metric table + confusion matrix + 5 failure examples + demo app | by 15:00 |
| **Tier 3: Wow (pick ONE)** | Confidence-based "needs human review" flag · Grad-CAM/explainability overlay · live webcam mode · batch report export (CSV/PDF) · latency benchmark CPU vs GPU · measurement/geometry output · tracking across frames | Works reliably in the demo | by 16:00 |

Rule: **never start Tier 3 before Tier 2 is demoable.** If Tier 2 is shaky at 16:00, drop Tier 3 entirely.

### A5. Pivot rules (decide with data, not hope)

- **By 12:00 no working prediction?** Drop fine-tuning. Use a pretrained model or a classical OpenCV baseline and move on.
- **Baseline metric clearly poor by 12:30?** Do failure analysis (Part G) before changing anything. Fix the most common failure mode only.
- **Training run not converging after ~45 min?** Stop it, shrink image size/subset, or switch to frozen-backbone transfer learning.
- **Labels look wrong/inconsistent?** Tell a mentor immediately; do not silently train on bad labels.
- **Dependency hell for >20 min?** Switch library/approach. Environment fights are the #1 time sink.

---

## Part B. Constraints and ground rules

- **Team:** 2 people.
- **Event:** one day; reporting 9:00 AM, expected end around 6:00 PM (confirm official schedule).
- **Data/task:** unknown until the venue. Inspect before choosing an algorithm.
- **Hardware:** plan for a laptop GPU with 4–8 GB VRAM; keep a CPU-compatible fallback.
- **Goal:** a working end-to-end solution, measurable results, a clear demo, and a compelling story. Not the largest model.

### Non-negotiable sequence

1. Read the task **and** the success criteria together.
2. Inspect sample images, labels, class names, dataset size.
3. Classify the problem: classification, detection, segmentation, OCR, counting, anomaly detection, retrieval, measurement, video.
4. Run a baseline as early as possible.
5. Improve only if the baseline works and time remains.
6. Keep the pipeline lightweight and demonstrable.

---

## Part C. Recognise the task type

| Task | Expected output | Example |
|---|---|---|
| Image classification | One or more labels per image | "Nozzle type A", "defective" |
| Object detection | Bounding box + class + confidence per object | Find every nozzle/part in the image |
| Semantic segmentation | Class per pixel | Spray / part / background regions |
| Instance segmentation | Separate mask per object | Mask each component |
| Anomaly / defect detection | Normal/abnormal score or defect location | Unusual surface or spray pattern |
| Object counting | Count, optionally with boxes | Components on a tray |
| OCR | Text, serial numbers, labels | Read a nozzle ID |
| Similarity / retrieval | Nearest reference image | Compare sample to approved reference |
| Measurement | Size, angle, area (needs calibration) | Spray angle, part dimensions |
| Enhancement | Better image quality | Denoise, contrast correction |
| Multi-image / video | Track or compare across frames | Follow an object through a sequence |

**"Detecting and identifying objects" can mean detection, classification, or both.** Detection finds *where*; identification names *what*. A strong design is often **two stages**: detect/crop, then classify the crop (fine-grained variants are easier to separate on a tight crop).

### Fast decision tree

- One label per image → **Classification**
- Multiple objects with locations → **Detection** (YOLO nano/small)
- Pixel-accurate regions needed → **Segmentation**
- Only "good" examples, defects unknown → **Anomaly detection** (embedding distance)
- Need a count → **Detection + counting**
- Need to read text → **OCR**
- Compare against known references → **Embeddings + nearest neighbour**
- Video/live camera → **Efficient inference + frame loop (+ tracking)**
- Controlled background/lighting, simple shapes → **Try classical OpenCV first**

### Questions for mentors (ask in the first 30 minutes)

1. What is the exact expected output format?
2. Classification, detection, segmentation, anomaly, or something else?
3. How many classes/objects? How many images?
4. Are labels provided and verified? Which annotation format?
5. Is there a train/validation/test split? Is the test set hidden?
6. Can we use pretrained models, open-source libraries, external data, cloud/GPU?
7. Is latency or "real-time" part of scoring? How is it defined?
8. **What are the judging criteria and their weights?**
9. Can we document assumptions?
10. What must be delivered: code, demo, slides, report, or all? What is the presentation length?
11. What does a costly mistake look like in practice (missed defect vs false alarm)?

Question 11 is the secret weapon: the answer tells you which metric to optimise.

---

## Part D. Dataset inspection (first 30 minutes after briefing)

### Checklist

- [ ] Number of images, dimensions, aspect ratios
- [ ] Colour format (RGB/BGR/grayscale)
- [ ] Quality: blur, glare, shadows, low light, noise
- [ ] Class names and **class balance**
- [ ] Label availability and format
- [ ] Annotation quality (missing boxes/masks, inconsistent labels)
- [ ] Train/val/test split present?
- [ ] **Duplicates / near-duplicates**
- [ ] **Leakage risk** (same object or scene in multiple splits)
- [ ] Corrupt files and broken paths
- [ ] Small enough for rapid iteration?

### Label formats

- **Classification:** CSV, folder names, JSON
- **Detection:** COCO JSON, Pascal VOC XML, YOLO `.txt`
- **Segmentation:** COCO polygons/RLE, PNG masks
- **Custom:** CSV/JSON/XML/spreadsheet

Do not convert or re-label the whole dataset until the required format is confirmed.

### Do these three things in the first hour

1. **Plot a contact sheet** of ~25 random images with labels drawn on. Most data problems show up here.
2. **Plot class counts.** Imbalance changes your metric and your augmentation plan.
3. **Create a leakage-safe split** *before* any training. If images come from the same part, session, or sequence, split by group:

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["part_or_session_id"]))
```

Fix the random seed and keep a **frozen "demo set" of unseen images** you never train on. Your live demo and final metrics come from it.

---

## Part E. Model selection and baselines

### Golden rule: baseline in under an hour

Run the simplest thing that could work, record its numbers, then improve. Your baseline is also your safety net if everything else fails.

### Classification

- Backbones: MobileNetV3, EfficientNet-B0/B1, ResNet18. A small ViT only if resources allow.
- Recipe: freeze the backbone → train the head → unfreeze top layers with a low learning rate only if time allows → evaluate on held-out data.
- For **fine-grained, many-variant** problems, also try **embeddings + kNN** against reference images. It needs no training and handles new variants by just adding references.

### Object detection

- YOLO nano/small variants for speed and low VRAM (use the latest nano weights your installed Ultralytics version provides).
- Verify annotation format, **visualise labels over images before training**, run pretrained inference on a few images first, then fine-tune.
- Minimal skeleton:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")            # or the nano weights available to you
model.train(data="data.yaml", imgsz=640, epochs=40, batch=8, device=0,
            patience=10, project="runs", name="baseline")
metrics = model.val()                 # mAP50, mAP50-95, precision, recall
```

- If the GPU runs out of memory: lower `batch`, then `imgsz` (e.g. 416 or 320).
- **Licence note:** Ultralytics YOLO is AGPL-3.0 (commercial licence otherwise). Fine for a hackathon; mention in "next steps" that a production deployment needs a licensing decision or an alternative such as an Apache/MIT-licensed detector.

### Segmentation

Expensive. Confirm it is truly required. Options: YOLO-seg nano/small, a lightweight pretrained segmenter, or classical thresholding + contours on controlled backgrounds.

### Anomaly / defect detection (strong when defects are rare)

Clarify first: defect *classification*, defect *localisation*, or *unknown anomaly* detection.

A fast, training-free baseline: extract embeddings from a pretrained CNN for "good" images, then score new images by distance to the nearest good embedding.

```python
import torch, torchvision, numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image

weights = torchvision.models.ResNet18_Weights.DEFAULT
net = torchvision.models.resnet18(weights=weights); net.fc = torch.nn.Identity(); net.eval()
prep = weights.transforms()

@torch.no_grad()
def embed(paths):
    x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths])
    return net(x).numpy()

good = embed(good_paths)
nn = NearestNeighbors(n_neighbors=1).fit(good)
score = nn.kneighbors(embed([test_path]))[0][0, 0]   # higher = more anomalous
# choose the threshold on a validation set, not by eye
```

### OCR

Only if text reading is part of the task. Crop, rotate, deskew first. Validate outputs, because glare and industrial fonts break OCR.

### Classical OpenCV: don't dismiss it

Fast, explainable, CPU-only, and sometimes beats deep learning on controlled setups. Useful for: CLAHE contrast fixing, Otsu/adaptive thresholds, morphological clean-up, contours with area/perimeter filters, Hough circles/lines, HSV colour masks, connected components, and simple measurements. Warning: brittle when lighting, background, or camera vary. Test on many images, never one "perfect" sample.

**A hybrid often wins:** classical preprocessing/cropping + a small learned classifier.

---

## Part F. Training playbook (low-resource)

- Prefer pretrained weights; start small (nano/tiny).
- Test inference **before** training.
- Verify the whole pipeline on a tiny subset first.
- Start at modest resolution (320/416); raise only if small objects are missed and memory allows.
- Batch size 1–4 to start; reduce on CUDA OOM.
- Short first run with early stopping; save checkpoints and **log every setting** (a small `experiments.csv` is enough).
- Don't run GPU-heavy jobs on both laptops against shared resources; use the second laptop for UI, evaluation, and slides while the first trains.
- **Use training time productively:** while a job runs, build the UI, write the pitch, and generate failure galleries from the previous checkpoint.
- If training is too slow: freeze more layers, shrink the subset/resolution, or skip fine-tuning if pretrained inference is good enough.
- RTX 3050 / GTX 1650 notes: avoid large models and batches, keep a CPU fallback, monitor VRAM (`nvidia-smi`), load one large model at a time, and remember the desktop uses some VRAM.

### Preprocessing and augmentation

- Resize preserving aspect ratio; letterbox rather than distort when geometry matters.
- Identical preprocessing at train and inference time.
- Augment only with physically meaningful transforms: small rotation, brightness/contrast, mild blur/noise, scale/crop, colour jitter. **Flips/rotations can be invalid** if orientation carries meaning.
- Small dataset? Heavier augmentation plus transfer learning. Few examples of one class? Class weights or oversampling.
- Check image–mask alignment for segmentation.

---

## Part G. Evaluation and failure analysis (where credibility is earned)

### Choose the metric that matches the cost

| Task | Report |
|---|---|
| Classification | Accuracy, precision, recall, F1, **confusion matrix**, per-class results (use macro-F1 when imbalanced) |
| Detection | Precision, recall, IoU, mAP@0.5 and mAP@0.5:0.95, false positives/negatives, confidence threshold |
| Segmentation | IoU, Dice, per-class metrics (pixel accuracy misleads with big backgrounds) |
| Anomaly | ROC-AUC/PR-AUC, false-alarm rate, localisation quality, normal vs anomalous examples |
| Performance | Latency per image, FPS, CPU/GPU and memory, end-to-end time incl. preprocessing |

**Never claim a metric you did not compute.** If no labelled validation data exists, say the results are qualitative and describe your small test set.

### Show the threshold trade-off

Plot precision vs recall (or ROC) across confidence thresholds and **choose the operating point from the business cost**: in inspection a missed defect usually costs more than a false alarm, so favour recall and route uncertain cases to a human. Saying this out loud is a strong signal to judges.

### Failure-analysis routine (30 minutes, after the baseline)

1. Export the 20–30 worst predictions (highest-confidence errors first).
2. Group them by cause: too small, blurred/occluded, lighting/glare, label noise, look-alike classes, background clutter, threshold too high/low, train/test mismatch.
3. Count each group; **fix the biggest group first** (more augmentation, higher resolution for small objects, relabel, threshold change, extra crops/tiling, class weights).
4. Re-measure on the same validation set and record the delta.
5. Keep the before/after as a slide: *"Baseline X → improved Y by doing Z."*

### Optional credibility boosters

- **Ablation table:** baseline vs + augmentation vs + higher resolution vs + two-stage.
- **Explainability:** Grad-CAM overlays (e.g. `pytorch-grad-cam`) to show the model looks at the part, not the background.
- **Confidence calibration / "review needed" band** for borderline predictions.
- **Robustness check:** run the demo set with brightness shift, blur, or rotation and report the drop.

---

## Part H. Architecture, demo, and presentation

### Pipeline

`Image input → preprocessing → model inference → postprocessing → structured result + annotated image`

### The interface contract (agree in the first hour)

Both teammates code against this single function so work never blocks:

```python
def predict(image_path: str) -> dict:
    """Return a JSON-serialisable result."""
    return {
        "detections": [
            {"label": "nozzle_A", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
        ],
        "count": 1,
        "needs_review": False,          # True when confidence is in the uncertain band
        "latency_ms": 23.4,
        "annotated_path": "outputs/img_001_annotated.jpg",
    }
```

Stub it with fake output on minute one so the UI can be built immediately; swap in the real model later.

### Demo options (pick the simplest that works)

- **Gradio or Streamlit** app: upload image → annotated result + table + latency. Fastest route to a convincing demo.
- CLI script or notebook as the fallback.
- A tiny FastAPI endpoint only if an API is explicitly valued.

Nice demo touches: confidence slider, side-by-side original/annotated, batch folder mode with a downloadable CSV, a "known failure" example shown honestly. **Do not over-invest in UI**, but do keep it clean.

### Suggested repo layout

```
project/
├── data/            # raw + splits (not committed if large)
├── notebooks/       # exploration only
├── src/
│   ├── predict.py   # the contract above
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── app.py           # Gradio/Streamlit demo
├── demo_samples/    # frozen unseen images + backup outputs
├── results/         # metrics, plots, failure gallery
├── requirements.txt
└── README.md        # 1-page: problem, approach, how to run, results
```

### 3-minute live-demo script

1. (15 s) One-sentence problem and who benefits.
2. (45 s) Upload a clean unseen image → annotated result, confidence, latency.
3. (45 s) Upload a hard image (glare/small/occluded) → show it still works or is flagged for review.
4. (30 s) Show the metric table and confusion matrix/PR curve.
5. (30 s) Show one honest failure and what you'd do about it.
6. (15 s) Close with the industrial impact and next step.

Always keep a **pre-recorded screen capture** and **saved output images** as a backup in case the live run fails.

### Presentation structure (5–7 minutes)

1. **Problem:** the industrial pain point and its cost (time, scrap, missed defects).
2. **Data:** what we received and its challenges.
3. **Approach:** why this method; what we rejected and why.
4. **Pipeline:** one clear architecture diagram.
5. **Results:** metric table, confusion matrix/PR curve, before→after improvement.
6. **Demo:** live, on unseen images.
7. **Limitations & failure analysis:** what breaks, why, how we'd fix it.
8. **Impact & next steps:** deployment path (edge/CPU, camera setup, more data, active learning, retraining loop, licence check), expected value.

Speaker split: one person owns problem/impact/story, the other owns the technical and demo sections; both rehearse the full flow twice.

---

## Part I. Two-person work split

### Person A — Model / Computer Vision
Inspect labels, choose baseline, preprocessing, train/infer, evaluate, error analysis, expose `predict()`.

### Person B — Data / Integration / Demo
Dataset visualisation and splits, repo hygiene, UI/API, integration, end-to-end testing, slides and demo narration, backup recordings.

### Shared
Agree on task and success criteria, review the baseline together at ~11:00, decide scope tier at 10:30 and 15:00, test the final solution, rehearse.

### Working rules
- Commit small and often to a shared Git repo (or at minimum copy checkpoints to a shared folder).
- One person owns each file to avoid merge conflicts.
- Don't wait on unfinished code: use the stub `predict()`.
- Hourly 2-minute sync: "what's done, what's blocked, are we on tier?"

---

## Part J. One-day time plan (adjust to the official agenda)

| Time | Target |
|---|---|
| 9:00–9:30 | Briefing: clarify goal, **scoring rubric**, constraints, deliverables |
| 9:30–10:15 | Inspect data, labels, contact sheet, class counts, **group-safe split**, freeze demo set |
| 10:15–10:30 | Decide task type, tier scope, baseline; agree `predict()` contract |
| 10:30–12:00 | Working baseline + stub UI connected |
| 12:00–12:30 | First metrics + failure analysis |
| 12:30–13:30 | Lunch / progress checkpoint (decide pivots) |
| 13:30–15:00 | Improve: fix top failure mode, tune threshold, fine-tune if justified |
| 15:00–16:00 | Tier 2 demo app finished; optionally one Tier 3 feature |
| 16:00–16:45 | Edge-case testing, latency benchmark, bug fixes, README |
| 16:45–17:30 | Slides, demo script, backup recording, rehearsal ×2 |
| 17:30–18:00 | Buffer / final polish |

### Milestones
- **10:30:** task understood, data inspected, baseline chosen
- **12:00:** first working output end-to-end
- **15:00:** improvement done, Tier 2 demoable
- **16:00:** **feature freeze**
- **17:00:** code freeze; only bug fixes and rehearsal

---

## Part K. Classical OpenCV quick reference

- **Basics:** load/save, BGR↔RGB, resize/crop/rotate/flip, normalise, grayscale, draw boxes/contours/masks.
- **Enhancement:** Gaussian/median/bilateral blur, histogram equalisation, CLAHE, careful sharpening, erosion/dilation/opening/closing.
- **Thresholding:** fixed, adaptive, Otsu.
- **Edges/shapes:** Canny, contours, area/perimeter, bounding rectangles, convex hull, Hough lines/circles.
- **Colour:** HSV conversion and thresholds, histograms, connected components.

---

## Part L. Common pitfalls (avoid these)

- Training before inspecting the data.
- Random train/val split when images are near-duplicates (inflated, fake accuracy).
- Reporting accuracy on imbalanced data.
- Testing the demo on training images.
- BGR/RGB mix-ups between OpenCV and PyTorch/PIL.
- Different preprocessing at training and inference.
- Flipping/rotating when orientation matters.
- Changing five things at once, so you can't tell what helped.
- Spending the afternoon on UI polish while the model is unevaluated.
- Last-hour risky changes, or a demo that depends on internet/cloud.
- Claiming results you can't reproduce.

---

## Part M. Anticipated judge questions (prepare short answers)

1. Why this model instead of a bigger one? *(speed, VRAM, data size, deployability)*
2. How do you know it isn't overfitting or leaking? *(group split, frozen unseen demo set)*
3. What happens on images unlike the training set? *(robustness test, review-needed flag)*
4. Which errors matter most and how did you tune for them? *(cost-based threshold)*
5. Could this run in production? *(latency numbers, CPU fallback, camera/lighting constraints, licence)*
6. How would it improve with more data/time? *(active learning, hard-negative mining, retraining loop, more variants)*
7. What did you try that didn't work? *(shows honest experimentation)*

---

## Part N. Pre-event preparation (do this *before* the day)

### Environment
Core: Python (a version already working on the laptop), VS Code, Git, NumPy, Pandas, Matplotlib, OpenCV (`opencv-python`), scikit-learn, Pillow.
Likely needed: PyTorch (correct CUDA build), Ultralytics, Gradio or Streamlit.
Optional: TensorFlow/Keras, OCR package, `pytorch-grad-cam`, segmentation packages.
Confirm every package **imports and runs a tiny example**; keep a known-good environment (e.g. exported `requirements.txt` / conda env) as backup. Don't install everything blindly at the last minute.

### Dry run (highly recommended, 2–3 hours)
Rehearse the entire loop on a public dataset, such as a small defect/inspection dataset (surface defects, MVTec-AD-style) or any small object-detection set: inspect → split → baseline → metrics → failure gallery → Gradio demo → 3-minute pitch. The goal is to have **reusable code** for the split, evaluation, failure gallery, `predict()` wrapper, and app on hackathon morning.

### Reusable starter kit to have ready
- Contact-sheet/label-viewer script
- Class-count plot
- Group-safe split helper
- Evaluation script (confusion matrix, PR curve, per-class table)
- Failure-gallery exporter
- `predict()` stub + Gradio app template
- Slide template with the 8-section structure
- Latency benchmark snippet (warm-up, then average over N runs)

### Logistics
Charged laptops, chargers and extension cord, offline copies of pretrained weights, a USB stick with backups, phone hotspot as an internet fallback.

---

## Part O. Final checklists

### Before starting
- [ ] Task, expected output, and **judging criteria** understood
- [ ] Allowed models/tools/data confirmed
- [ ] Dataset inspected, samples visualised, split made, demo set frozen
- [ ] GPU/CPU and environment checked
- [ ] Baseline chosen, tier scope decided
- [ ] `predict()` contract agreed

### Before the final demo
- [ ] End-to-end pipeline works from input to output
- [ ] Tested on many unseen examples, not just one
- [ ] Metrics computed honestly and reproducibly
- [ ] Failure analysis and a before→after improvement ready
- [ ] Latency measured (CPU and GPU if available)
- [ ] Model/checkpoint saved; README explains how to run
- [ ] Demo works offline; backup images and screen recording ready
- [ ] Limitations stated honestly; next steps defined
- [ ] Pitch rehearsed at least twice, with timing
- [ ] No risky changes after code freeze

---

## Final principle

**Dataset first. Task second. Baseline third. Evidence always. Story throughout.**

The winning one-day solution is usually a lightweight, well-evaluated system with a reliable demo, an honest failure analysis, and a clear line from the model's output to industrial value, not a large model that barely finishes training.
