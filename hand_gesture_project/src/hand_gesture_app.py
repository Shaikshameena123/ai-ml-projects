import cv2
import mediapipe as mp
from tkinter import *
from threading import Thread

from .gesture_recognition import is_thumbs_up

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")
        self.root.geometry("800x600")

        self.label_var = StringVar()
        self.label = Label(self.root, textvariable=self.label_var, font=("Helvetica", 24))
        self.label.pack()

        self.start_button = Button(self.root, text="Start Hand Tracking", command=self.start_hand_tracking)
        self.start_button.pack()

        self.stop_button = Button(self.root, text="Stop Hand Tracking", command=self.stop_hand_tracking)
        self.stop_button.pack()

        self.cap = None
        self.tracking = False
        self.tracking_thread = None

    def start_hand_tracking(self):
        if self.tracking:
            return
        
        self.tracking = True
        self.cap = cv2.VideoCapture(0)
        self.tracking_thread = Thread(target=self.track_hands)
        self.tracking_thread.start()

    def track_hands(self):
        while self.tracking and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark

                    if is_thumbs_up(landmarks):
                        self.label_var.set("Thumbs Up Detected")
                    else:
                        self.label_var.set("")

            cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                self.stop_hand_tracking()

        self.cap.release()
        cv2.destroyAllWindows()

    def stop_hand_tracking(self):
        self.tracking = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.tracking_thread:
            self.tracking_thread.join()

    def run(self):
        self.root.mainloop()
