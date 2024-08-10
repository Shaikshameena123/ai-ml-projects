import cv2

def draw_landmarks(frame, pose_landmarks, mp_pose, mp_drawing):
    mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

def detect_gesture(frame, pose_landmarks, mp_pose):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    if left_wrist.y < 0.5 and right_wrist.y < 0.5:
        cv2.putText(frame, 'Hands are raised!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
