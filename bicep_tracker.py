import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 pose estimation model
model = YOLO("yolov8n-pose.pt")

def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Mid point (e.g., elbow)
    c = np.array(c)  # End point (e.g., wrist)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate the cosine of the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    # Ensure the value is within the valid range for arccos
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert angle to degrees
    return np.degrees(angle)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Bicep curl counter variables
counter = 0 
stage = None  # Can be 'down' or 'up'

print("Starting real-time bicep curl tracking...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Get pose detection results from the YOLO model
    # verbose=False hides the detailed model output
    results = model(frame, verbose=False)
    
    # Plot the pose landmarks and connections on the frame
    annotated_frame = results[0].plot()

    # --- Bicep Curl Logic ---
    try:
        # Extract keypoints from the results
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Get coordinates for right arm landmarks
        # Indices are based on the COCO keypoints format used by YOLO
        right_shoulder = keypoints[6]
        right_elbow = keypoints[8]
        right_wrist = keypoints[10]
        
        # Check if keypoints are detected (coordinates are not [0, 0])
        if all(coord > 0 for coord in np.concatenate([right_shoulder, right_elbow, right_wrist])):
            # Calculate the angle of the right elbow
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # --- Curl Counting Logic ---
            # If the arm is extended (angle > 160), set stage to 'down'
            if angle > 160:
                stage = "down"
            # If the arm is flexed (angle < 40) and was previously 'down', 
            # it's the 'up' phase. Increment counter.
            if angle < 40 and stage == 'down':
                stage = "up"
                counter += 1
                print(f"Repetition Count: {counter}")

            # --- Display Information on Frame ---
            # Create a semi-transparent rectangle for better text visibility
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 130), (20, 20, 20), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # Display Repetition Counter
            cv2.putText(annotated_frame, 'REPS', (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, str(counter), (40, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Display Curl Stage
            cv2.putText(annotated_frame, 'STAGE', (200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, stage if stage else 'START', (210, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

            # Display Elbow Angle (for debugging or information)
            cv2.putText(annotated_frame, f'Angle: {int(angle)}', (30, annotated_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # If keypoints are not detected, show a message
            cv2.putText(annotated_frame, "Position your right arm in the frame", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    except Exception as e:
        # This will catch errors if no person is detected in the frame
        # and prevent the program from crashing.
        cv2.putText(annotated_frame, "No person detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the final annotated frame
    cv2.imshow('Bicep Curl Tracker - Real Time', annotated_frame)
    
    # Handle key presses
    # Wait for 1 millisecond, and check if the pressed key is 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print(f"\nSession ended. Total reps: {counter}")
cap.release()
cv2.destroyAllWindows()
