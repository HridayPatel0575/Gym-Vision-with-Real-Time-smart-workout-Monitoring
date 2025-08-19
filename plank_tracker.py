import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO

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

# 🎯 Ask for target seconds
target_secs = int(input("Enter your plank target (seconds): "))

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

log = []
good_start_time = None
held_seconds = 0
success = False

print("Starting real-time plank detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)[0]
    keypoints = results.keypoints

    if keypoints and len(keypoints.xy[0]) > 0:
        kpts = keypoints.xy[0].cpu().numpy()
        if len(kpts) > 14:
            right_shoulder, right_hip, right_knee = kpts[6], kpts[12], kpts[14]
            if all(p[0] > 0 and p[1] > 0 for p in [right_shoulder, right_hip, right_knee]):
                hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                plank_quality = "Good Plank" if 160 <= hip_angle <= 195 else "Bad Plank"

                annotated = results.plot()

                if plank_quality == "Good Plank":
                    if good_start_time is None:
                        good_start_time = time.time()
                    else:
                        held_seconds = int(time.time() - good_start_time)

                    if held_seconds >= target_secs:
                        success = True
                        cv2.putText(annotated, "🎉 You did it!", (150, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                        cv2.imshow('Plank Trainer', annotated)
                        cv2.waitKey(5000)  # Show for 5 sec
                        break
                else:
                    good_start_time = None
                    held_seconds = 0

                cv2.putText(annotated, f'Held: {held_seconds}/{target_secs} sec', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.putText(annotated, plank_quality, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 255, 0) if "Good" in plank_quality else (0, 0, 255), 3)
                cv2.imshow('Plank Trainer', annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ✅ After camera closes
if success:
    print(f"You held the plank for {target_secs} seconds! Well done.")
else:
    print("Try Again!")
