# Powerlifting Barbell Tracker
### YOLOv8 · OpenCV · MediaPipe · Velocity-Based RPE Estimation

A computer vision system that tracks barbell movement in powerlifting videos,
calculates bar velocity using real-world scale calibration, estimates training
intensity (RPE) from velocity, and counts reps — for three lifts: deadlift,
bench press, and squat.

Built as part of a Computer Vision course project at Manipal Institute of
Technology. Custom dataset collected from personal gym footage and labelled
manually.

---

## Features

**All three lifts:**
- Click-based plate calibration — maps pixels to metres using a 45cm plate as reference
- Bar path drawn as a live trajectory overlay on the video
- Per-rep velocity in m/s using frame timestamps + calibrated scale
- RPE estimation from last-rep velocity (Velocity-Based Training methodology)

**Deadlift (`final_deadlift_v2.py`):**
- Two-pass architecture: backend scan locks the floor/lockout height thresholds
  from the first rep so counting works correctly from frame 1
- Pure YOLO inference per frame — barbell is large and clearly visible

**Bench Press (`final_bench_v2.py`):**
- YOLO + CSRT hybrid: YOLO re-detects every 5 frames, CSRT handles
  fast horizontal motion between detections
- Backend scan (first 200 frames) pre-initialises height thresholds
  before the main loop, fixing the "first rep never counted" bug

**Squat (`squats_v2.py`):**
- Dual-sensor: MediaPipe Pose for depth/rep detection, YOLO + CSRT for bar path
- Depth detected via knee angle (< 100°) — more reliable than hip-below-knee
  pixel comparison across different camera distances and angles
- Configurable unracking skip (`SKIP_SECONDS`) to ignore the walk-out
  before any rep counting begins
- Live state debug overlay: `IN SQUAT / RISING / STANDING` + frame counter

---

## RPE from Velocity (Velocity-Based Training)

| Deadlift | Bench | Squat | Est. RPE |
|----------|-------|-------|----------|
| ≥ 0.70 m/s | ≥ 0.45 m/s | ≥ 0.50 m/s | RPE 6 |
| ≥ 0.60 m/s | ≥ 0.35 m/s | ≥ 0.40 m/s | RPE 7 |
| ≥ 0.50 m/s | ≥ 0.25 m/s | ≥ 0.30 m/s | RPE 8 |
| ≥ 0.40 m/s | ≥ 0.15 m/s | ≥ 0.20 m/s | RPE 9 |
| < 0.40 m/s | < 0.15 m/s | < 0.20 m/s | RPE 10 |

*Based on: Sánchez-Medina & González-Badillo (2011) and VBT research.*

---

## How Calibration Works

On startup, the first frame is displayed. Click on two points along the edge
of a 45cm competition plate. This maps pixels to real-world metres:

```
px_to_m  = 0.45 / distance_between_clicks_in_pixels
velocity = (bar_displacement_px × px_to_m) / rep_time_seconds
```

The same calibration window is used in all three scripts.

---

## Dataset

Custom dataset built from personal gym training footage, labelled manually
using Roboflow.

| Split | Images |
|-------|--------|
| Train | ~700   |
| Val   | ~170   |
| **Total** | **~870** |

- **Single class:** `barbell_end` (the circular plate end of the barbell)
- **Format:** YOLO `.txt` bounding box annotations
- **Base model:** YOLOv8n (nano — chosen for inference speed)
- **Training:** 200 epochs, 640px image size, GPU

**Best model performance (train17):**

| Metric | Value |
|--------|-------|
| mAP@50 | 0.847 |
| mAP@50-95 | **0.913** |

---

## Known Limitations

**Dataset size and generalisation:**
The training dataset has ~870 images collected from a single gym environment.
The model works well on similar footage but may struggle with different gyms,
lighting conditions, or barbell styles. A larger, more diverse dataset would
significantly improve robustness.

**Playback speed:**
Running YOLO + CSRT + MediaPipe simultaneously produces lower-than-realtime
frame rates on mid-range hardware. The video will appear slower than the
original. Using a dedicated GPU or reducing inference resolution helps.

**Velocity accuracy:**
Velocity is calculated from vertical bar displacement between the start and
end of the concentric phase. If the tracker loses the barbell mid-rep, the
velocity reading for that rep will be incorrect. The squat tracker prints
a console warning when this happens.

**Single plate assumption:**
Calibration assumes a standard 45cm diameter plate is visible in frame.
Update `PLATE_DIAMETER_M` in the config block if using smaller plates.

---

## Setup

```bash
git clone https://github.com/sahityajain360/barbell-tracker
cd barbell-tracker
pip install -r requirements.txt
```

GPU recommended. Change `model.to("cuda")` to `model.to("cpu")` if needed.

Place your video in the `videos/` folder and update `VIDEO_PATH` in the
config block at the top of each script.

```bash
python final_deadlift_v2.py
python final_bench_v2.py
python squats_v2.py
```

At startup: click two points on the plate edge to calibrate, then press
any key to begin tracking. Press `Q` to quit.

---

## Project Structure

```
barbell-tracker/
├── final_deadlift_v2.py    # Deadlift — two-pass YOLO
├── final_bench_v2.py       # Bench — YOLO + CSRT hybrid
├── squats_v2.py            # Squat — YOLO + MediaPipe
├── weights/
│   └── best.pt             # Trained YOLOv8n weights (~6MB)
├── dataset/
│   └── dataset.yaml        # Class config (barbell_end)
├── requirements.txt
└── README.md
```

---

## Future Work

- **Live camera mode** — real-time tracking via webcam or phone camera,
  enabling competition-style judging with voice commands ("Squat", "Press",
  "Rack") and automatic pass/fail verdicts per rep
- **Larger dataset** — more diverse gym environments and lighting to improve
  generalisation across different barbells and settings
- **Automatic scale detection** — estimate real-world scale from pose landmark
  proportions (shoulder width) to remove the manual calibration step
- **Processing speed** — frame skipping or async inference to achieve closer
  to real-time playback on mid-range hardware

---

## About

Built by Sahitya Rajeev Jain, B.Tech CSE (AI & ML) at Manipal Institute
of Technology (2023–27).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahitya360-blue)](https://linkedin.com/in/sahitya360/)
[![GitHub](https://img.shields.io/badge/GitHub-sahityajain360-black)](https://github.com/sahityajain360)
