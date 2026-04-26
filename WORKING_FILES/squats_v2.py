import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
import collections

# ============================================================
# CONFIGURATION — edit these before running
# ============================================================
VIDEO_PATH              = "videos/sample_squat11.mp4"
WEIGHTS_PATH            = "weights/best.pt"
PLATE_DIAMETER_M        = 0.45      # standard 45cm competition plate
VISIBLE_SIDE            = "right"   # "left" or "right" — whichever faces camera
MIN_FRAMES_IN_BOTTOM    = 3         # frames at depth before concentric triggers
LANDMARK_CONF_THRESHOLD = 0.6
DEPTH_SMOOTHING_WINDOW  = 5
MAX_PATH_LENGTH         = 20
SQUAT_ANGLE_THRESHOLD   = 100       # angle below this = "in the hole" (ignore unracks ~110-120°)
LOCKOUT_ANGLE_THRESHOLD = 145       # angle above this = locked out
SKIP_SECONDS            = 3         # ignore first N seconds (unracking period)


# ============================================================
# RPE ESTIMATION (Velocity-Based Training)
# Source: VBT research — squat velocity profiles
# ============================================================
def squat_velocity_to_rpe(velocity_mps: float) -> int:
    """
    Estimate RPE for squat based on mean concentric bar velocity (m/s).
    Squat thresholds sit between deadlift and bench press profiles.
    Source: VBT research (González-Badillo et al.)
    """
    if velocity_mps >= 0.50:   return 6
    elif velocity_mps >= 0.40: return 7
    elif velocity_mps >= 0.30: return 8
    elif velocity_mps >= 0.20: return 9
    else:                      return 10


# ============================================================
# CALIBRATION — click two points on a 45cm plate edge
# ============================================================
points       = []
scale_factor = None
px_to_m      = None

def click_event(event, x, y, flags, param):
    global points, frame_resized, scale_factor, px_to_m
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration: Click plate top & bottom edge", frame_resized)

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            dist_px_resized  = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
            dist_px_original = dist_px_resized / scale_factor
            px_to_m          = PLATE_DIAMETER_M / dist_px_original
            print(f"✅ Scale calculated: {px_to_m:.6f} m/px")
            cv2.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Calibration: Click plate top & bottom edge", frame_resized)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video for calibration.")

max_w, max_h = 960, 540
h0, w0       = first_frame.shape[:2]
scale_factor = min(max_w / w0, max_h / h0)
new_w, new_h = int(w0 * scale_factor), int(h0 * scale_factor)
frame_resized = cv2.resize(first_frame, (new_w, new_h))

cv2.imshow("Calibration: Click plate top & bottom edge", frame_resized)
cv2.setMouseCallback("Calibration: Click plate top & bottom edge", click_event)
print("Click on the top and bottom edges of a 45cm plate, then press any key.")
cv2.waitKey(0)
cv2.destroyWindow("Calibration: Click plate top & bottom edge")

if px_to_m is None:
    raise RuntimeError("Calibration failed. Please click two points on the plate.")

cap.release()


# ============================================================
# MODEL + MEDIAPIPE SETUP
# ============================================================
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose()

yolo_model = YOLO(WEIGHTS_PATH)
yolo_model.to("cuda")

# MediaPipe landmark indices based on visible side
if VISIBLE_SIDE == "right":
    HIP_IDX      = mp_pose.PoseLandmark.RIGHT_HIP.value
    KNEE_IDX     = mp_pose.PoseLandmark.RIGHT_KNEE.value
    ANKLE_IDX    = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    SHOULDER_IDX = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
else:
    HIP_IDX      = mp_pose.PoseLandmark.LEFT_HIP.value
    KNEE_IDX     = mp_pose.PoseLandmark.LEFT_KNEE.value
    ANKLE_IDX    = mp_pose.PoseLandmark.LEFT_ANKLE.value
    SHOULDER_IDX = mp_pose.PoseLandmark.LEFT_SHOULDER.value


# ============================================================
# HELPERS
# ============================================================
def calculate_angle(a, b, c):
    """Returns the angle at joint b formed by points a-b-c (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def adjust_landmark(landmark, ref1, ref2, offset_ratio):
    """Nudges a landmark slightly toward a reference direction for better accuracy."""
    dx = ref2[0] - ref1[0]
    dy = ref2[1] - ref1[1]
    dist = np.sqrt(dx**2 + dy**2)
    return (landmark[0], landmark[1] - offset_ratio * dist)


# ============================================================
# TRACKING STATE
# ============================================================
cap        = cv2.VideoCapture(VIDEO_PATH)
last_time  = time.time()
frame_count = 0

# Rep counting (MediaPipe-driven)
rep_count         = 0
frames_in_bottom  = 0
rep_in_progress   = False

# Velocity tracking (bar Y from YOLO at rep start/end)
rep_start_frame   = None
rep_start_bar_y   = None   # bar Y pixel when concentric begins
last_rep_velocity = None

# YOLO + CSRT bar tracking
tracking  = False
tracker   = None
bar_path  = []
current_bar_y = None       # updated every frame from YOLO/CSRT


# ============================================================
# MAIN LOOP
# ============================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    # Skip unracking period at start of video
    if frame_count < SKIP_SECONDS * fps_video:
        cv2.putText(frame, f"Skipping unrack... {SKIP_SECONDS - frame_count/fps_video:.1f}s",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.imshow("Squat Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # FPS
    now      = time.time()
    fps_proc = 1.0 / max(now - last_time, 1e-6)
    last_time = now
    cv2.putText(frame, f"FPS: {fps_proc:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ── YOLO + CSRT Barbell Tracking ────────────────────────
    success = False
    box = None

    if not tracking or frame_count % 10 == 0:
        # Resize for lighter YOLO inference (avoids GPU crash with MediaPipe)
        resized_w = 640
        resized_h = int(h * resized_w / w)
        small_frame = cv2.resize(frame, (resized_w, resized_h))

        detections = yolo_model(small_frame, conf=0.5, verbose=False)[0]
        best_box, best_conf = None, 0
        for det in detections.boxes:
            conf = float(det.conf[0])
            if conf > best_conf:
                # Scale coordinates back to original resolution
                sx, sy = w / resized_w, h / resized_h
                x1_r, y1_r, x2_r, y2_r = map(int, det.xyxy[0].tolist())
                x1 = int(x1_r * sx); y1 = int(y1_r * sy)
                x2 = int(x2_r * sx); y2 = int(y2_r * sy)
                best_box = (x1, y1, x2 - x1, y2 - y1)
                best_conf = conf
        if best_box is not None:
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, best_box)
            tracking = True
            success, box = True, best_box
            if not bar_path: bar_path = []
    else:
        if tracking:
            success, box = tracker.update(frame)

    if success:
        x, y_box, bw, bh = [int(v) for v in box]
        center = (x + bw // 2, y_box + bh // 2)
        current_bar_y = center[1]              # ← used for velocity

        bar_path.append(center)
        if len(bar_path) > MAX_PATH_LENGTH:
            bar_path.pop(0)

        cv2.rectangle(frame, (x, y_box), (x + bw, y_box + bh), (0, 255, 0), 2)
        for i in range(1, len(bar_path)):
            cv2.line(frame, bar_path[i - 1], bar_path[i], (0, 255, 255), 3)
    else:
        tracking      = False
        tracker       = None
        bar_path      = []
        current_bar_y = None

    # ── MediaPipe Pose ──────────────────────────────────────
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = pose.process(rgb_frame)

    if mp_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = mp_results.pose_landmarks.landmark

        # Only process if landmarks are confident
        if (lm[HIP_IDX].visibility > LANDMARK_CONF_THRESHOLD and
                lm[KNEE_IDX].visibility > LANDMARK_CONF_THRESHOLD):

            hip      = (lm[HIP_IDX].x,      lm[HIP_IDX].y)
            knee     = (lm[KNEE_IDX].x,     lm[KNEE_IDX].y)
            ankle    = (lm[ANKLE_IDX].x,    lm[ANKLE_IDX].y)
            shoulder = (lm[SHOULDER_IDX].x, lm[SHOULDER_IDX].y)

            adj_hip  = adjust_landmark(hip,  shoulder, knee,  0.05)
            adj_knee = adjust_landmark(knee, hip,      ankle, 0.03)

            angle = calculate_angle(hip, knee, ankle)

            # --- Simple angle-based depth detection ---
            # Squat is "in the hole" when knee angle drops below 120°
            in_squat = angle < SQUAT_ANGLE_THRESHOLD

            if in_squat:
                frames_in_bottom += 1
            else:
                # Lifter is rising / standing — check transitions
                if frames_in_bottom >= MIN_FRAMES_IN_BOTTOM and not rep_in_progress:
                    # Was at depth long enough → concentric phase begins
                    rep_in_progress = True
                    rep_start_frame = frame_count
                    rep_start_bar_y = current_bar_y
                    print(f"[DEBUG] Concentric started at frame {frame_count}, "
                          f"bar_y={current_bar_y}, angle={angle:.1f}")

                # Lockout → rep complete (angle > 145°)
                if angle > LOCKOUT_ANGLE_THRESHOLD and rep_in_progress:
                    rep_count      += 1
                    rep_in_progress = False

                    # Velocity calculation
                    if (rep_start_frame is not None and
                            current_bar_y is not None and
                            rep_start_bar_y is not None):
                        rep_time = (frame_count - rep_start_frame) / fps_video
                        if rep_time > 0:
                            displacement_px = abs(rep_start_bar_y - current_bar_y)
                            displacement_m  = displacement_px * px_to_m
                            last_rep_velocity = displacement_m / rep_time
                            print(f"✅ Rep {rep_count}: {displacement_m:.3f} m in "
                                  f"{rep_time:.2f} s → {last_rep_velocity:.3f} m/s "
                                  f"→ RPE {squat_velocity_to_rpe(last_rep_velocity)}")
                    else:
                        print(f"✅ Rep {rep_count}: (no bar velocity — YOLO lost track)")

                    rep_start_frame = None
                    rep_start_bar_y = None

                frames_in_bottom = 0

            # Draw knee angle
            knee_px = tuple(np.multiply(knee, [w, h]).astype(int))
            cv2.putText(frame, str(int(angle)), knee_px,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Debug overlay — shows what the state machine sees
            state_str = "IN SQUAT" if in_squat else ("RISING" if rep_in_progress else "STANDING")
            color = (0, 165, 255) if in_squat else ((0, 255, 255) if rep_in_progress else (0, 255, 0))
            cv2.putText(frame, f"State: {state_str}  |  Bottom frames: {frames_in_bottom}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Rep display (outside pose block so it always shows)
    cv2.putText(frame, f"Squat Reps: {rep_count}", (10, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # ── Velocity & RPE Display ───────────────────────────────
    if last_rep_velocity is not None:
        est_rpe = squat_velocity_to_rpe(last_rep_velocity)
        cv2.putText(frame, f"Last Rep Vel: {last_rep_velocity:.2f} m/s", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Est. RPE: {est_rpe}", (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Squat Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()