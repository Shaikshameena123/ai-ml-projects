import cv2
import mediapipe as mp
from utils import draw_landmarks, detect_gesture

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Global variable to control the detection process
is_running = False

def start_pose_detection():
    global is_running
    is_running = True
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            draw_landmarks(frame, results.pose_landmarks, mp_pose, mp_drawing)
            detect_gesture(frame, results.pose_landmarks, mp_pose)

        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_pose_detection():
    global is_running
    is_running = False
    print("Pose detection stopped.")
