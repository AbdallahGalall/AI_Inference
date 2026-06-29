# Workout Pose Analysis — Graduation Project

A computer vision pipeline for real-time **workout form correction** and **pose analysis** using deep learning. The project combines YOLO-based 2D pose estimation, MediaPipe 3D pose tracking, body orientation classification, and few-shot exercise recognition into a unified fitness coaching system.

---

## Repository Structure

```
.
├── yolopose.ipynb          # YOLO11 pose estimation & form analysis
├── mediapip-pose.ipynb     # MediaPipe 3D pose reference generation
├── person-view.ipynb       # Body orientation classifier (front/side/back)
└── one-few-shot.ipynb      # One/few-shot exercise recognition with EfficientNet
```

---

## Notebooks

### 1. `yolopose.ipynb` — YOLO11 Pose Estimation & Form Correction

Real-time workout form analysis using **YOLO11n-pose** to detect 2D keypoints and evaluate exercise technique across multiple movements.

**Exercises covered:** Bicep Curl, Hammer Curl, Tricep Pushdown, Pushup, Plank, Squat, Deadlift

**Key metrics computed per exercise:**

| Metric | Description |
|--------|-------------|
| Elbow-body ratio | `dist(elbow, hip) / dist(shoulder, hip)` — detects elbow flare |
| Trunk lean | Horizontal shoulder-hip deviation — detects forward/backward lean |
| Hip alignment | Deviation from shoulder-ankle line — detects sag or pike in plank/pushup |
| Knee width ratio | Knee spread relative to hip width — detects knee collapse |
| Back angle | Hip angle between shoulder → hip → knee vectors |
| Elbow angle | Joint angle at elbow for curl exercises |
| Elbow depth offset | Forward/backward elbow position relative to shoulder |

**Features:**
- Percentile-based adaptive thresholds calibrated per exercise
- Phase detection (TOP / MID / BOTTOM) for squat and deadlift
- Debounced violation banners drawn on annotated video output
- Reference JSON generation from "good form" video samples
- Confidence-based side selection (left vs. right keypoints per frame)

**Dependencies:** `ultralytics`, `opencv-python`, `numpy`

---

### 2. `mediapip-pose.ipynb` — MediaPipe 3D Pose Reference Generation

Extends the YOLO pipeline with true **3D pose estimation** using **MediaPipe Pose**, enabling camera-angle-robust form analysis.

**Why MediaPipe instead of YOLO for references?**

YOLO outputs 2D `(x, y)` keypoints — all measurements are perspective-dependent. MediaPipe outputs `(x, y, z)` in metric-scale world coordinates, enabling true 3D angles and distances regardless of camera angle.

**Improvements over 2D pipeline:**

| Feature | YOLO (2D) | MediaPipe (3D) |
|--------|-----------|----------------|
| Distance | Euclidean `√(dx²+dy²)` | 3D Euclidean `√(dx²+dy²+dz²)` |
| Angles | 2D dot product | 3D dot product on `(x,y,z)` vectors |
| Trunk lean | X-axis pixel offset | Z-axis depth `(shoulder_z - hip_z) / torso_3d` |
| Hip alignment | 2D point-line distance | 3D point-line distance |
| Knee collapse | 2D width ratio | 3D knee/ankle width ratio |

**Keypoint mapping (YOLO COCO → MediaPipe):**

```
YOLO idx  →  MediaPipe landmark
   5       →  11  LEFT_SHOULDER
   6       →  12  RIGHT_SHOULDER
   7       →  13  LEFT_ELBOW
  11       →  23  LEFT_HIP
  13       →  25  LEFT_KNEE
  15       →  27  LEFT_ANKLE
```

**Graceful fallback:** If `z` coordinates are all-zero, the system automatically falls back to 2D mode. Output JSON structure is identical to the YOLO notebook for drop-in compatibility.

**Dependencies:** `mediapipe==0.10.14`, `opencv-python-headless`, `numpy`, `pillow`

---

### 3. `person-view.ipynb` — Body Orientation Classifier

Classifies whether a person in a video frame is facing **Front**, **Side**, or **Back** relative to the camera. Used to automatically select the correct form-checking metrics for each camera angle.

**Dataset:** [Human3.6M](http://vision.imar.ro/human3.6m/) — 3D joint positions + RGB videos across subjects S1, S5, S6, S7, S8 (train) and S9 (test). Actions: Walking, WalkingDog, Standing, Directions.

**Pipeline:**

```
3D Joint Data (.cdf)
      │
      ▼
Hip vector → Yaw angle → Orientation label (Front / Side / Back)
      │
      ▼
RGB frame crop (via joint bounding box)
      │
      ▼
Balanced dataset: 5,000 images × 3 classes = 15,000 images
      │
      ▼
MobileNetV3-Small (fine-tuned, ImageNet weights)
      │
      ▼
Confusion matrix + Cross-subject evaluation (S9)
```

**Model:** MobileNetV3-Small — chosen for its lightweight inference suitable for real-time use alongside the pose estimation pipeline.

**Configuration:**

```python
IMAGES_PER_CLASS = 5000     # balanced dataset
IMG_SIZE         = 224
BATCH_SIZE       = 32
EPOCHS           = 20
LR               = 1e-3
```

> **Mock data mode:** Set `USE_MOCK_DATA = True` to run and test the full pipeline without the Human3.6M dataset.

**Output:** `orientation_classifier.pth` — PyTorch model checkpoint

**Dependencies:** `torch`, `torchvision`, `spacepy` (for `.cdf` files), `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

---

### 4. `one-few-shot.ipynb` — Food Recognition for Nutrition & Macros

A **metric-learning** approach to food image classification using **EfficientNetB2** as a backbone with triplet loss. Identifies food items from a photo to enable automatic macro tracking and nutritional surveillance — users can log meals by simply taking a picture.

**Dataset:** [Food-101](https://www.kaggle.com/datasets/kmader/food41) — 101 food categories, ~1,000 images per class.

**Architecture:**

```
Input Image (224×224)
      │
      ▼
EfficientNetB2 (backbone, pretrained)
      │
      ▼
Embedding Head → 128-dim L2-normalized vector
      │
      ▼
Triplet Loss (margin = 0.5)
      │
      ▼
Support Set Prototypes (mean embedding per class)
      │
      ▼
Nearest-prototype classification
```

**Training strategy:**

| Phase | Epochs | LR | Notes |
|-------|--------|----|-------|
| Phase 1 | 15 | 1e-3 | Backbone frozen, head only |
| Phase 2 | 25 | 1e-5 | Full fine-tune, lower LR |

**Triplet sampling:** For each class, `N` triplets are sampled as `(anchor, positive, negative)` where positive is a different image of the same class and negative is from a random different class.

```python
N_TRAIN_TRIPLETS   = 200    # per class  → ~20,200 total
N_TEST_TRIPLETS    = 50
N_SHOTS_PER_CLASS  = 10     # images used to build each prototype
EMBEDDING_DIM      = 128
DROPOUT_RATE       = 0.4
```

**Output:** `support_set.pkl` — serialized food prototype embeddings. At inference time, a photo is encoded and matched to the nearest prototype to identify the food item, which can then be looked up in a nutrition database for calorie and macro breakdown.

**Dependencies:** `tensorflow`, `opencv-python`, `numpy`, `scikit-learn`, `seaborn`, `kagglehub`, `tf_slim`

---

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback supported)

### Install dependencies

```bash
# YOLO pipeline
pip install ultralytics opencv-python numpy

# MediaPipe pipeline
pip install mediapipe==0.10.14 opencv-python-headless numpy pillow

# Orientation classifier
pip install torch torchvision spacepy opencv-python scikit-learn matplotlib seaborn tqdm

# Few-shot recognition
pip install tensorflow tf-slim kagglehub scikit-learn seaborn
```

---

## How the Notebooks Connect

```
person-view.ipynb
      │
      │  Camera angle → Front / Side / Back
      ▼
yolopose.ipynb
      │
      │  2D keypoints + form metrics + violation feedback
      ▼
mediapip-pose.ipynb
      │
      │  3D keypoints + camera-robust reference thresholds
      │  (same JSON output format → same corrector logic)
      ▼
   Workout Feedback / Form Corrector


one-few-shot.ipynb  (independent module)
      │
      │  Photo of meal → Food identity (few-shot, no retraining)
      ▼
   Nutrition DB lookup → Calories / Macros / Nutritional surveillance
```

---

## Key Design Decisions

**Percentile-based thresholds** — Rather than hard-coding angle limits, thresholds are derived from a reference set of correct-form videos per exercise, making the system self-calibrating to different body types and camera setups.

**Debounced violation detection** — Violations must persist for multiple consecutive frames before a banner is displayed, reducing false positives from momentary pose estimation noise.

**MediaPipe as reference generator, YOLO for real-time** — YOLO runs fast enough for real-time feedback; MediaPipe generates the gold-standard reference JSON used to calibrate thresholds, combining the strengths of both models.

**Few-shot for food recognition** — Using metric learning means new food items can be added to the system by providing a small support set (10 images), without retraining the backbone. This makes the nutrition module extensible to regional cuisines or custom meal types not present in Food-101.

---

## License

This project was developed as part of a graduation project. See individual dataset licenses (Food-101, Human3.6M, WorkoutFitness-Video on Kaggle) for data usage terms.
