import cv2
import numpy as np
import pandas as pd
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

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Log for storing pose data
log = []

print("Starting real-time pose detection...")
print("Press 'q' to quit, 's' to save log")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Get pose results
    results = model(frame, verbose=False)[0]
    keypoints = results.keypoints
    
    if keypoints and len(keypoints.xy[0]) > 0:
        kpts = keypoints.xy[0].cpu().numpy()
        
        # Check if we have enough keypoints
        if len(kpts) > 14:
            right_shoulder = kpts[6]
            right_hip = kpts[12]
            right_knee = kpts[14]
            
            # Only calculate if all keypoints are detected
            if all(point[0] > 0 and point[1] > 0 for point in [right_shoulder, right_hip, right_knee]):
                hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                
                # Pose classification based on hip angle
                plank_quality = "Good Plank" if 160 <= hip_angle <= 195 else "Bad Plank"
                
                # Plot annotations
                annotated = results.plot()
                
                # Add text overlay
                cv2.putText(annotated, f'Hip Angle: {int(hip_angle)} deg', (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.putText(annotated, plank_quality, (30, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, 
                           (0, 255, 0) if "Good" in plank_quality else (0, 0, 255), 3)
                
                # Add instructions
                cv2.putText(annotated, "Press 'q' to quit, 's' to save log", (30, annotated.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Log the data
                log.append({
                    "hip_angle": hip_angle,
                    "pose_quality": plank_quality
                })
                
                # Display the frame
                cv2.imshow('Gym Position Training - Real Time', annotated)
            else:
                # Display frame without pose analysis if keypoints not detected
                cv2.putText(frame, "Position yourself in frame", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", (30, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Gym Position Training - Real Time', frame)
    else:
        # Display frame without pose analysis if no keypoints detected
        cv2.putText(frame, "No pose detected", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (30, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Gym Position Training - Real Time', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if log:
            df = pd.DataFrame(log)
            df.to_csv('pose_training_log.csv', index=False)
            print(f"Log saved! Total entries: {len(log)}")
        else:
            print("No data to save")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save final log
if log:
    df = pd.DataFrame(log)
    df.to_csv('pose_training_log.csv', index=False)
    print(f"Final log saved with {len(log)} entries")
    print(f"Average hip angle: {df['hip_angle'].mean():.2f} degrees")
    good_planks = len(df[df['pose_quality'] == 'Good Plank'])
    print(f"Good planks: {good_planks}/{len(log)} ({good_planks/len(log)*100:.1f}%)")
else:
    print("No pose data recorded")
