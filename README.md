# Powerlifting Barbell Tracker
### YOLOv8 · OpenCV · MediaPipe · Velocity-Based RPE Estimation

A computer vision system that tracks barbell movement in powerlifting videos,
calculates bar velocity using real-world scale calibration, estimates training
intensity (RPE) from velocity, and counts reps — for three lifts: deadlift,
bench press, and squat.

Built as part of a Computer Vision course project at Manipal Institute of
Technology. Custom dataset collected from personal gym footage.

---

## Demo

| Deadlift | Bench Press | Squat |
|----------|-------------|-------|
| Bar path + velocity + RPE | YOLO+CSRT hybrid | MediaPipe depth detection |

---

## Features

**All three lifts:**
- Real-world scale calibration via click-based plate measurement (45cm reference)
- Bar path drawn as a trajectory overlay on the video
- Per-rep velocity calculation in m/s using frame timestamps and calibrated scale
- RPE estimation from last-rep velocity using VBT (Velocity-Based Training) research

**Deadlift (`final_deadlift_v2.py`):**
- Two-pass architecture: backend scan locks min/max bar height from the first rep,
  so rep counting works correctly from frame 1
- Pure YOLO inference — barbell is large and clearly visible in deadlifts

**Bench Press (`final_bench_v2.py`):**
- YOLO + CSRT hybrid tracker: YOLO re-detects every 5 frames for accuracy,
  CSRT tracker handles fast horizontal motion between YOLO frames
- Backend scan (first 200 frames) initializes height thresholds before tracking begins

**Squat (`squats_v2.py`):**
- Dual-sensor system: MediaPipe Pose for depth detection, YOLO for bar path
- Depth validated by hip-below-knee criterion and knee angle (< 90°)
- Moving average smoothing on landmark positions to reduce MediaPipe jitter
- Rep counted only after confirmed depth + return to lockout (angle > 160°)

---

## How It Works

### 1. Calibration
On startup, the first frame is displayed. Click on two points along the edge
of a 45cm competition plate. This maps pixels to metres for velocity calculation.

```
px_to_m = 0.45 / distance_in_pixels
velocity = displacement_m / rep_time_s
```

### 2. RPE from Velocity (VBT)

| Deadlift Velocity | Bench Velocity | Estimated RPE |
|-------------------|----------------|---------------|
| ≥ 0.70 m/s        | ≥ 0.45 m/s     | RPE 6–7       |
| ≥ 0.60 m/s        | ≥ 0.35 m/s     | RPE 7–8       |
| ≥ 0.50 m/s        | ≥ 0.25 m/s     | RPE 8–9       |
| ≥ 0.40 m/s        | ≥ 0.15 m/s     | RPE 9–9.5     |
| < 0.40 m/s        | < 0.15 m/s     | RPE 10        |

*Source: Sánchez-Medina & González-Badillo (2011), VBT research.*

### 3. Rep Counting

**Deadlift / Bench:**
Thresholds set from backend scan. A rep is counted when the bar travels
from the bottom zone (floor / chest) through the full range to the
top zone (lockout), with a minimum displacement to filter noise.

**Squat:**
MediaPipe detects the right hip and knee landmarks. A rep is counted when
the hip drops below the knee (depth confirmed for ≥ 10 frames) and then
the lifter returns to full lockout (knee angle > 160°).

---

## Dataset

Custom dataset built from personal gym training footage:

| Split | Images |
|-------|--------|
| Train | ~2,400 |
| Val   | ~600   |
| **Total** | **~3,000** |

- **Single class:** `barbell_end` (the circular plate end of the barbell)
- **Format:** YOLO `.txt` annotation files
- **Training:** YOLOv8n, 200 epochs, 640px image size

**Best model performance (train17):**

| Metric | Value |
|--------|-------|
| mAP@50 | 0.847 |
| mAP@50-95 | **0.913** |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/sahityajain360/barbell-tracker
cd barbell-tracker

# Install dependencies
pip install -r requirements.txt

# Place your video in the project root and update VIDEO_PATH in the script
```

**Requirements:**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
```

GPU recommended (CUDA). The scripts call `model.to("cuda")` — change to
`model.to("cpu")` if running on CPU only.

---

## Usage

Each script has a `CONFIG` block at the top. Change `VIDEO_PATH` to point
to your video before running.

```bash
# Deadlift
python final_deadlift_v2.py

# Bench Press
python final_bench_v2.py

# Squat
python squats_v2.py
```

**At startup:** A window shows the first frame. Click on the top and bottom
edges of a 45cm plate to calibrate. Press any key to begin tracking.

**During tracking:** Press `Q` to quit.

---

## Project Structure

```
barbell-tracker/
├── final_deadlift_v2.py    # Deadlift tracker (two-pass, pure YOLO)
├── final_bench_v2.py       # Bench tracker (YOLO + CSRT hybrid)
├── squats_v2.py            # Squat tracker (YOLO + MediaPipe)
├── weights/
│   └── best.pt             # Trained YOLOv8n weights (~6MB)
├── dataset/
│   └── dataset.yaml        # Class config (barbell_end)
├── requirements.txt
└── README.md
```

---

## Future Work

- **Live camera mode** — real-time tracking via webcam or phone camera feed,
  enabling a competition-style judging system with voice commands
  ("Squat", "Press", "Rack") and automatic white/red light verdicts
- **Scale without calibration** — estimate px_to_m automatically using
  pose landmark proportions (shoulder width as reference)
- **Multi-barbell tracking** — handle training partners in the same frame
- **Web interface** — upload a video and receive an annotated output file

---

## About

Built by Sahitya Rajeev Jain, B.Tech CSE (AI & ML) at Manipal Institute
of Technology (2023–27).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahitya360-blue)](https://linkedin.com/in/sahitya360/)
[![GitHub](https://img.shields.io/badge/GitHub-sahityajain360-black)](https://github.com/sahityajain360)
