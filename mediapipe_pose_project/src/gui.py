import tkinter as tk
from threading import Thread
from pose_detection import start_pose_detection, stop_pose_detection

# Global thread variable
pose_thread = None

def start_thread():
    global pose_thread
    if pose_thread is None or not pose_thread.is_alive():
        pose_thread = Thread(target=start_pose_detection)
        pose_thread.start()

def stop_thread():
    stop_pose_detection()
    if pose_thread is not None:
        pose_thread.join()

def setup_gui():
    root = tk.Tk()
    root.title("MediaPipe Pose Detection")

    start_button = tk.Button(root, text="Start", command=start_thread)
    start_button.pack()

    stop_button = tk.Button(root, text="Stop", command=stop_thread)
    stop_button.pack()

    root.mainloop()

if __name__ == "__main__":
    setup_gui()
