# Barbell Tracker Project - Standalone Working Files

This directory contains the standalone, ready-to-run versions of the Powerlifting Barbell Tracker scripts. These scripts have been extracted from research notebooks and enhanced with a competition-style judging system and asynchronous voice commands.

## 📂 File Structure

```text
WORKING_FILES/
├── final_deadlift_v2.py  # Standalone Deadlift Tracker
├── final_bench_v2.py     # Standalone Bench Press Tracker (YOLO + CSRT)
├── squats_v2.py          # Standalone Squat Tracker (YOLO + MediaPipe)
├── videos/               # Sample videos for testing
│   ├── deadlift2.mp4
│   ├── bench2.mp4
│   └── sample_squat6.mp4
└── weights/              # Trained YOLOv8 model weights
    └── best.pt
```

## ⚙️ How the System Works

### 1. Deadlift Tracker (`final_deadlift_v2.py`)
- **Two-Pass Architecture**:
  - **Pass 1 (Backend)**: Scans the video once to establish the floor and lockout heights. This ensures rep counting works from the very first frame.
  - **Pass 2 (Frontend)**: Runs the tracking with the locked thresholds, calculating velocity and RPE.
- **Judging Criteria**:
  - **Hitching Detection**: Detects if the bar moves downward by more than 10px during the upward phase.
  - **Velocity Check**: Validates that the lift was performed with a minimum average velocity (0.15 m/s).

### 2. Bench Press Tracker (`final_bench_v2.py`)
- **Hybrid Tracking**: Combines YOLOv8 (accuracy) with CSRT (smoothness). YOLO runs every 5 frames to re-center the tracker, while CSRT handles the high-speed motion between frames.
- **Backend Scan**: Performs a quick initial scan (first 200 frames) to set height thresholds, fixing the "first rep detection" bug.
- **Judging Criteria**:
  - **Touch Confirmation**: Ensures the bar reached the bottom zone (chest).
  - **Press Command**: Detects if the lifter pressed the bar before it reached the chest (press early).

### 3. Squat Tracker (`squats_v2.py`)
- **Dual-Sensor System**: Uses MediaPipe Pose for biomechanical depth detection (hip vs knee) and YOLO for barbell path tracking.
- **State Machine**: Tracks the lift through states: `waiting_unrack` -> `waiting_start` -> `rep_in_progress` -> `waiting_rack`.
- **Judging Criteria**:
  - **Depth**: Validates that the hip crease went below the top of the knee.
  - **Lockout**: Ensures the lifter returned to a full upright position (>160 degrees).

## 🚀 Key Features

- **Async Voice Judge**: Uses `threading` and `pyttsx3` to provide voice commands ("Squat", "Press", "Rack", "White Light") in a background thread. This ensures the video loop never stutters or blocks.
- **Visual Verdicts**: Displays "WHITE LIGHT" (Pass) or "RED LIGHT" (Fail with reason) on the screen for 3 seconds after each rep.
- **Calibration**: A simple click-based calibration tool is built into every script to map pixels to real-world meters using a standard 45cm plate as a reference.

## 🛠 Setup & Usage

1. **Install Requirements**:
   ```bash
   pip install opencv-python numpy ultralytics pyttsx3 mediapipe nbformat
   ```
2. **Run a Script**:
   ```bash
   python final_deadlift_v2.py
   ```
3. **Quit**: Press **'q'** at any time to exit the tracking window.
