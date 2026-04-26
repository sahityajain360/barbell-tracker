import cv2
import numpy as np
from ultralytics import YOLO
import time
VIDEO_PATH   = "videos/bench2.mp4"
WEIGHTS_PATH = "weights/best.pt"
PLATE_DIAMETER_M  = 0.45
CONF_THRESHOLD    = 0.5
YOLO_EVERY_N      = 5              # run YOLO every N frames
MAX_PATH_LENGTH   = 30
TOP_ZONE_PCT      = 0.20
BOTTOM_ZONE_PCT   = 0.20
MIN_RANGE_PX      = 20

def bench_velocity_to_rpe(velocity_mps: float) -> float:
    """
    Estimate RPE for bench press based on last-rep bar velocity (m/s).
    Source: VBT research (Sánchez-Medina & González-Badillo, 2011, etc.)
    """
    if velocity_mps >= 0.45: return 6
    elif velocity_mps >= 0.35: return 7
    elif velocity_mps >= 0.25: return 8
    elif velocity_mps >= 0.15: return 9
    else: return 10.0

plate_diameter_real = PLATE_DIAMETER_M
points = []
scale = None
px_to_m = None

def click_event(event, x, y, flags, param):
    global points, frame_resized, scale, px_to_m
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click Plate Diameter", frame_resized)

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            dist_px_resized = np.linalg.norm(np.array([x2-x1, y2-y1]))
            dist_px_original = dist_px_resized / scale
            px_to_m = plate_diameter_real / dist_px_original

            print(f"👉 Plate diameter in resized frame: {dist_px_resized:.1f} px")
            print(f"👉 Plate diameter in original frame: {dist_px_original:.1f} px")
            print(f"👉 Scale: {px_to_m:.6f} m/px")

            cv2.line(frame_resized, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.imshow("Click Plate Diameter", frame_resized)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

ret, frame = cap.read()
if not ret: raise RuntimeError("Could not read video for calibration")

max_w, max_h = 960, 540
h, w = frame.shape[:2]
scale = min(max_w / w, max_h / h)
new_w, new_h = int(w * scale), int(h * scale)
frame_resized = cv2.resize(frame, (new_w, new_h))

cv2.imshow("Click Plate Diameter", frame_resized)
cv2.setMouseCallback("Click Plate Diameter", click_event)
cv2.waitKey(0)
cv2.destroyWindow("Click Plate Diameter")

if px_to_m is None: raise RuntimeError("Calibration not done.")
cap.release()

print("🔍 Running backend pass to lock initial thresholds...")
model = YOLO(WEIGHTS_PATH)
model.to("cuda")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
bar_min_y, bar_max_y = None, None

while cap.isOpened() and frame_count < 200:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    if frame_count % 2 == 0:
        detections = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        if len(detections.boxes) == 0: continue
        best_box, best_conf = None, 0
        for det in detections.boxes:
            conf = float(det.conf[0])
            if conf > best_conf:
                best_conf = conf
                y1, y2 = int(det.xyxy[0][1]), int(det.xyxy[0][3])
                bar_y = (y1 + y2) // 2
        
        if bar_min_y is None:
            bar_min_y, bar_max_y = bar_y, bar_y
        bar_min_y = min(bar_min_y, bar_y)
        bar_max_y = max(bar_max_y, bar_y)

locked_min_y, locked_max_y = bar_min_y, bar_max_y
print(f"Backend pass done. min_y: {locked_min_y}, max_y: {locked_max_y}")
cap.release()

tracker = None
tracking = False
bar_path = []

rep_count = 0
in_bottom_position = False
frame_count = 0
rep_start_frame = None
rep_start_y = None
last_rep_velocity = None

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    frame_count += 1
    success, box = False, None

    if not tracking or frame_count % YOLO_EVERY_N == 0:
        detections = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        best_box, best_conf = None, 0
        for det in detections.boxes:
            conf = float(det.conf[0])
            if conf > best_conf:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                bw, bh = x2 - x1, y2 - y1
                best_box = (x1, y1, bw, bh)
                best_conf = conf
        if best_box is not None:
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, best_box)
            tracking = True
            success, box = True, best_box
            if not bar_path: bar_path = []
    else:
        if tracking: success, box = tracker.update(frame)

    if success:
        x, y, bw, bh = [int(v) for v in box]
        center = (x + bw//2, y + bh//2)
        bar_path.append(center)
        if len(bar_path) > MAX_PATH_LENGTH: bar_path.pop(0)

        cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
        for i in range(1, len(bar_path)):
            cv2.line(frame, bar_path[i-1], bar_path[i], (0,255,255), 3)

        bar_y = center[1]
        
        if locked_min_y is None:
            bar_min_y = bar_y
            bar_max_y = bar_y
            locked_min_y, locked_max_y = bar_y, bar_y
        
        bar_min_y = min(locked_min_y, bar_y) 
        bar_max_y = max(locked_max_y, bar_y)
        locked_min_y, locked_max_y = bar_min_y, bar_max_y

        range_y = bar_max_y - bar_min_y
        if range_y > MIN_RANGE_PX:
            top_threshold = bar_min_y + TOP_ZONE_PCT * range_y
            bottom_threshold = bar_max_y - BOTTOM_ZONE_PCT * range_y

            if bar_y > bottom_threshold and not in_bottom_position:
                in_bottom_position = True
                rep_start_frame = None

            if in_bottom_position and bar_y < bottom_threshold:
                rep_start_frame = frame_count
                rep_start_y = bar_y
                in_bottom_position = False

            if rep_start_frame is not None and bar_y < top_threshold:
                rep_end_frame = frame_count
                rep_time = (rep_end_frame - rep_start_frame) / fps
                if rep_time > 0:
                    displacement_m = abs((rep_start_y - bar_y) * px_to_m)
                    avg_velocity = displacement_m / rep_time
                    last_rep_velocity = avg_velocity
                rep_count += 1
                rep_start_frame = None
    else:
        tracking = False

    cv2.putText(frame, f"Bench Reps: {rep_count}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    if last_rep_velocity is not None:
        est_rpe = bench_velocity_to_rpe(last_rep_velocity)
        cv2.putText(frame, f"Last Rep Vel: {last_rep_velocity:.2f} m/s", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f"Est. RPE: {est_rpe}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    max_w, max_h = 960, 540
    h_frame, w_frame = frame.shape[:2]
    scale = min(max_w / w_frame, max_h / h_frame)
    new_w, new_h = int(w_frame * scale), int(h_frame * scale)
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow("Bench Press Tracking", frame_resized)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()