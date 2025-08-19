import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLOv8 pose estimation model
model = YOLO("yolov8n-pose.pt")

def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# Ask for today's target
target_reps = int(input("Enter your target reps for today: "))

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables
counter = 0
stage = None

print(f"Starting real-time bicep curl tracking... Target = {target_reps} reps")
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
        right_shoulder = keypoints[6]
        right_elbow = keypoints[8]
        right_wrist = keypoints[10]

        if all(coord > 0 for coord in np.concatenate([right_shoulder, right_elbow, right_wrist])):
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if angle > 160:
                stage = "down"
            if angle < 40 and stage == 'down':
                stage = "up"
                counter += 1
                print(f"Repetition Count: {counter}")

            # Display Counter & Target
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 150), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            cv2.putText(annotated_frame, f'Target: {target_reps}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f'Count: {counter}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

            # If target reached
            if counter >= target_reps and not target_reached:
                target_reached = True
                end_time = time.time() + 5  # 5 seconds timer
                print("🎉 You did it! 🎉")

        else:
            cv2.putText(annotated_frame, "Position your right arm in the frame", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    except:
        cv2.putText(annotated_frame, "No person detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # If target is reached, show "You did it!" on screen
    if target_reached:
        cv2.putText(annotated_frame, " YOU DID IT! ", (250, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
        if time.time() > end_time:  # End after 5 seconds
            break

    cv2.imshow('Bicep Curl Tracker - Real Time', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nSession ended. Total reps: {counter}")
cap.release()
cv2.destroyAllWindows()
