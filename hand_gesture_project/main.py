from src.hand_gesture_app import HandGestureApp
from tkinter import Tk

if __name__ == "__main__":
    root = Tk()
    app = HandGestureApp(root)
    app.run()
