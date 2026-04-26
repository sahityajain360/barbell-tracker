import cv2
import numpy as np
from ultralytics import YOLO
import time
VIDEO_PATH   = "videos/deadlift2.mp4"     # ← change this
WEIGHTS_PATH = "weights/best.pt" # ← relative path
PLATE_DIAMETER_M = 0.45            # 45cm standard plate
CONF_THRESHOLD   = 0.5
MAX_PATH_LENGTH  = 30
TOP_ZONE_PCT     = 0.20            # top 20% = lockout zone
BOTTOM_ZONE_PCT  = 0.20            # bottom 20% = floor zone
MIN_RANGE_PX     = 20              # ignore noise below 20px

def deadlift_velocity_to_rpe(velocity_mps: float) -> float:
    """
    Estimate RPE for deadlift based on last-rep bar velocity (m/s).
    Source: VBT research (Sánchez-Medina & González-Badillo, 2011, etc.)
    """
    if velocity_mps >= 0.70: return 6.5
    elif velocity_mps >= 0.60: return 7.5
    elif velocity_mps >= 0.50: return 8.5
    elif velocity_mps >= 0.40: return 9.5
    else: return 10.0

# --------- Calibration Setup ---------
plate_diameter_real = PLATE_DIAMETER_M
points, scale, px_to_m = [], None, None

def click_event(event, x, y, flags, param):
    global points, frame_resized, scale, px_to_m
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click Plate Diameter", frame_resized)

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            dist_px_resized = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
            dist_px_original = dist_px_resized / scale
            px_to_m = plate_diameter_real / dist_px_original
            print(f"👉 Scale calculated: {px_to_m:.6f} m/px")
            cv2.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Click Plate Diameter", frame_resized)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

ret, frame = cap.read()
if not ret: raise RuntimeError("Could not read video for calibration")
max_w, max_h = 1280, 720
h, w = frame.shape[:2]
scale = min(max_w / w, max_h / h)
new_w, new_h = int(w * scale), int(h * scale)
frame_resized = cv2.resize(frame, (new_w, new_h))
cv2.imshow("Click Plate Diameter", frame_resized)
cv2.setMouseCallback("Click Plate Diameter", click_event)
cv2.waitKey(0)
cv2.destroyWindow("Click Plate Diameter")
if px_to_m is None: raise RuntimeError("Calibration failed.")
cap.release()

print("🔍 Backend scan running to prime state...")

model = YOLO(WEIGHTS_PATH)
model.to("cuda")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
bar_min_y, bar_max_y = None, None
in_bottom_position = False
rep_start_frame, rep_start_y = None, None
locked_min_y, locked_max_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    detections = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    if len(detections.boxes) == 0: continue

    det = detections.boxes[0]
    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
    bar_y = (y1 + y2) // 2

    if bar_min_y is None:
        bar_min_y, bar_max_y = bar_y, bar_y
    bar_min_y = min(bar_min_y, bar_y)
    bar_max_y = max(bar_max_y, bar_y)
    range_y = bar_max_y - bar_min_y

    if range_y > MIN_RANGE_PX:
        top_threshold = bar_min_y + TOP_ZONE_PCT * range_y
        bottom_threshold = bar_max_y - BOTTOM_ZONE_PCT * range_y

        if bar_y > bottom_threshold:
            in_bottom_position = True
        elif in_bottom_position:
            rep_start_frame = frame_count
            rep_start_y = bar_y
            in_bottom_position = False

        if rep_start_frame is not None and bar_y < top_threshold:
            rep_end_frame = frame_count
            rep_time = (rep_end_frame - rep_start_frame) / fps
            displacement_m = abs((rep_start_y - bar_y) * px_to_m)
            if rep_time > 0:
                velocity = displacement_m / rep_time
                rpe = deadlift_velocity_to_rpe(velocity)
                print(f"✅ Backend rep detected: Frames {rep_start_frame}-{rep_end_frame}, "
                      f"Disp={displacement_m:.3f} m, Time={rep_time:.3f} s, "
                      f"Vel={velocity:.3f} m/s, RPE={rpe}")
            locked_min_y, locked_max_y = bar_min_y, bar_max_y
            break

cap.release()

print("🎥 Starting main playback...")

cap = cv2.VideoCapture(VIDEO_PATH)
rep_count, last_rep_velocity, est_rpe = 0, None, None
frame_count = 0
in_bottom_position = False
rep_start_frame, rep_start_y = None, None
bar_path = []

rep_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    detections = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    if len(detections.boxes) == 0:
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Deadlift Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    det = detections.boxes[0]
    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
    center = (x1 + (x2 - x1)//2, y1 + (y2 - y1)//2)
    bar_y = center[1]

    # --- Bar path drawing ---
    bar_path.append(center)
    if len(bar_path) > MAX_PATH_LENGTH:
        bar_path.pop(0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i in range(1, len(bar_path)):
        cv2.line(frame, bar_path[i-1], bar_path[i], (0, 255, 255), 3)

    # --- Rep detection using locked thresholds ---
    if locked_min_y is not None and locked_max_y is not None:
        top_threshold = locked_min_y + TOP_ZONE_PCT * (locked_max_y - locked_min_y)
        bottom_threshold = locked_max_y - BOTTOM_ZONE_PCT * (locked_max_y - locked_min_y)

        if bar_y > bottom_threshold and not in_bottom_position:
            in_bottom_position = True
            rep_start_frame = None
            rep_active = False

        if in_bottom_position and bar_y < bottom_threshold:
            rep_start_frame = frame_count
            rep_start_y = bar_y
            in_bottom_position = False
            rep_active = True

        if rep_active and bar_y < top_threshold:
            rep_active = False
            rep_end_frame = frame_count
            rep_time = (rep_end_frame - rep_start_frame) / fps
            avg_velocity = 0
            if rep_time > 0:
                displacement_m = abs((rep_start_y - bar_y) * px_to_m)
                avg_velocity = displacement_m / rep_time
                last_rep_velocity = avg_velocity
                est_rpe = deadlift_velocity_to_rpe(avg_velocity)
                rep_count += 1
                print(f"Rep {rep_count}: Frames {rep_start_frame}-{rep_end_frame}, "
                      f"Disp={displacement_m:.3f} m, Time={rep_time:.3f} s, "
                      f"Vel={avg_velocity:.3f} m/s, RPE={est_rpe}")

            rep_start_frame = None

    # --------- Display ---------
    cv2.putText(frame, f"Deadlift Reps: {rep_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    if last_rep_velocity is not None:
        cv2.putText(frame, f"Last Rep Vel: {last_rep_velocity:.2f} m/s", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Est. RPE: {est_rpe}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # Resize for display
    max_w, max_h = 1280, 720
    h_frame, w_frame = frame.shape[:2]
    scale_disp = min(max_w / w_frame, max_h / h_frame)
    new_w_disp, new_h_disp = int(w_frame * scale_disp), int(h_frame * scale_disp)
    frame_resized = cv2.resize(frame, (new_w_disp, new_h_disp), interpolation=cv2.INTER_AREA)

    cv2.imshow("Deadlift Tracking", frame_resized)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()