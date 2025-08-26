import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

target_reps = int(input("Enter your target reps for today: "))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
stage = None
print(f"Starting real-time bicep curl tracking (BOTH HANDS)... Target = {target_reps} reps")
print("Press 'q' to quit early.")

target_reached = False
end_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    try:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        right_shoulder, right_elbow, right_wrist = keypoints[6], keypoints[8], keypoints[10]
        left_shoulder, left_elbow, left_wrist = keypoints[5], keypoints[7], keypoints[9]

        if all(coord > 0 for coord in np.concatenate([right_shoulder, right_elbow, right_wrist, 
                                                      left_shoulder, left_elbow, left_wrist])):
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_angle  = calculate_angle(left_shoulder,  left_elbow,  left_wrist)

            if right_angle > 160 and left_angle > 160:
                stage = "down"

            if right_angle < 40 and left_angle < 40 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Repetition Count: {counter}")

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (520, 180), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            cv2.putText(annotated_frame, f'Target: {target_reps}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f'Count: {counter}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            if counter >= target_reps and not target_reached:
                target_reached = True
                end_time = time.time() + 5
                print("You did it with BOTH arms!")

        else:
            cv2.putText(annotated_frame, "Position both arms in the frame", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    except:
        cv2.putText(annotated_frame, "No person detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if target_reached:
        cv2.putText(annotated_frame, "YOU DID IT!", (250, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
        if time.time() > end_time:
            break

    cv2.imshow('Bicep Curl Tracker - Both Hands', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nSession ended. Total reps: {counter}")
cap.release()
cv2.destroyAllWindows()
