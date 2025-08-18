import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def get_kpt(kpts, idx):
    if idx < len(kpts):
        return kpts[idx]
    return np.array([0.0, 0.0])


# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Squat counter
counter = 0
stage = None  # 'up' or 'down'
log = []

print("Starting real-time squat tracking...")
print("Press 'q' to quit, 's' to save CSV log")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to capture frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    try:
        kpts = results[0].keypoints.xy[0].cpu().numpy()

        # Prefer right side hip-knee-ankle; fallback to left
        r_hip = get_kpt(kpts, 12)
        r_knee = get_kpt(kpts, 14)
        r_ankle = get_kpt(kpts, 16)

        use_left = False
        if not all(coord > 0 for coord in np.concatenate([r_hip, r_knee, r_ankle])):
            l_hip = get_kpt(kpts, 11)
            l_knee = get_kpt(kpts, 13)
            l_ankle = get_kpt(kpts, 15)
            if all(coord > 0 for coord in np.concatenate([l_hip, l_knee, l_ankle])):
                use_left = True
                hip, knee, ankle = l_hip, l_knee, l_ankle
            else:
                hip, knee, ankle = r_hip, r_knee, r_ankle
        else:
            hip, knee, ankle = r_hip, r_knee, r_ankle

        if all(coord > 0 for coord in np.concatenate([hip, knee, ankle])):
            knee_angle = calculate_angle(hip, knee, ankle)

            # Counting logic: standing when knee ~ 170+, bottom when knee ~ < 90
            if knee_angle > 165:
                stage = 'up'
            if knee_angle < 95 and stage == 'up':
                stage = 'down'  # reached bottom
            if knee_angle > 150 and stage == 'down':
                stage = 'up'   # rose back up
                counter += 1

            # UI overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (420, 150), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            cv2.putText(annotated, 'SQUATS', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(annotated, f'REPS: {counter}', (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated, f'STAGE: {stage if stage else "START"}', (220, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(annotated, f'Knee Angle: {int(knee_angle)}', (20, annotated.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            log.append({
                'metric': 'knee_angle',
                'value': float(knee_angle),
                'stage': stage or 'start',
                'reps': int(counter)
            })
        else:
            cv2.putText(annotated, 'Show full legs in frame', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    except Exception:
        cv2.putText(annotated, 'No person detected', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.putText(annotated, "Press 'q' to quit, 's' to save log", (20, annotated.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Squat Tracker - Real Time', annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if log:
            df = pd.DataFrame(log)
            df.to_csv('squat_log.csv', index=False)
            print(f"Log saved! Entries: {len(log)}")
        else:
            print('No data to save')

print(f"\nSession ended. Total squats: {counter}")
cap.release()
cv2.destroyAllWindows()
