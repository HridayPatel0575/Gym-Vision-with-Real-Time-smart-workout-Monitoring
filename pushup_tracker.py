import cv2
import numpy as np
import pandas as pd
import time
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


# 🎯 Ask for target
target_reps = int(input("Enter your target push-ups for today: "))

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Push-up counter
counter = 0
stage = None  # 'up' or 'down'
log = []
target_reached = False
end_time = None

print(f"Starting real-time push-up tracking... Target = {target_reps} reps")
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

        # Right arm
        r_shoulder = get_kpt(kpts, 6)
        r_elbow = get_kpt(kpts, 8)
        r_wrist = get_kpt(kpts, 10)

        # Fallback to left if right not visible
        use_left = False
        if not all(coord > 0 for coord in np.concatenate([r_shoulder, r_elbow, r_wrist])):
            l_shoulder = get_kpt(kpts, 5)
            l_elbow = get_kpt(kpts, 7)
            l_wrist = get_kpt(kpts, 9)
            if all(coord > 0 for coord in np.concatenate([l_shoulder, l_elbow, l_wrist])):
                use_left = True
                shoulder, elbow, wrist = l_shoulder, l_elbow, l_wrist
            else:
                shoulder, elbow, wrist = r_shoulder, r_elbow, r_wrist
        else:
            shoulder, elbow, wrist = r_shoulder, r_elbow, r_wrist

        if all(coord > 0 for coord in np.concatenate([shoulder, elbow, wrist])):
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Counting logic
            if elbow_angle > 150:
                stage = 'up'
            if elbow_angle < 80 and stage == 'up':
                stage = 'down'
                counter += 1
                print(f"Repetition Count: {counter}")

            # UI overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (500, 150), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            cv2.putText(annotated, f'Target: {target_reps}', (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(annotated, f'REPS: {counter}', (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated, f'STAGE: {stage if stage else "START"}', (220, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(annotated, f'Elbow Angle: {int(elbow_angle)}', (20, annotated.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            log.append({
                'metric': 'elbow_angle',
                'value': float(elbow_angle),
                'stage': stage or 'start',
                'reps': int(counter)
            })

            # 🎉 Target reached logic
            if counter >= target_reps and not target_reached:
                target_reached = True
                end_time = time.time() + 5
                print("🎉 You did it! 🎉")

        else:
            cv2.putText(annotated, 'Make sure your arm is visible', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    except Exception:
        cv2.putText(annotated, 'No person detected', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show completion message if target reached
    if target_reached:
        cv2.putText(annotated, "🎉 YOU DID IT! 🎉", (250, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        if time.time() > end_time:
            break

    cv2.putText(annotated, "Press 'q' to quit, 's' to save log", (20, annotated.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Push-up Tracker - Real Time', annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not target_reached:
        if log:
            df = pd.DataFrame(log)
            df.to_csv('pushup_log.csv', index=False)
            print(f"Log saved! Entries: {len(log)}")
        else:
            print('No data to save')

print(f"\nSession ended. Total push-ups: {counter}")
cap.release()
cv2.destroyAllWindows()
