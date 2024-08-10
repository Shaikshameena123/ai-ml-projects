import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

class VQAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Question Answering")
        
        self.capture_btn = tk.Button(root, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack(pady=10)
        
        self.question_label = tk.Label(root, text="Enter your question:")
        self.question_label.pack()
        
        self.question_entry = tk.Entry(root, width=50)
        self.question_entry.pack(pady=10)
        
        self.ask_btn = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_btn.pack(pady=10)
        
        self.answer_label = tk.Label(root, text="")
        self.answer_label.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        # Load pre-trained model and processor
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        
    def capture_image(self):
        cap = cv2.VideoCapture(0)  # Open the default camera (0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video device")
            return
        
        ret, frame = cap.read()  # Read the frame
        cap.release()  # Release the capture
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return
        
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_image(self.image)
        
    def display_image(self, image):
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=image)
        self.image_label.image = image
    
    def preprocess_image(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return pil_image
    
    def ask_question(self):
        question = self.question_entry.get()
        
        if not hasattr(self, 'image'):
            messagebox.showerror("Error", "Please capture an image first")
            return
        
        preprocessed_image = self.preprocess_image(self.image)
        encoding = self.processor(preprocessed_image, question, return_tensors="pt")
        
        # Get the answer from the model
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Decode the answer
        logits = outputs.logits
        predicted_id = logits.argmax(-1).item()
        answer = self.model.config.id2label[predicted_id]
        
        self.answer_label.config(text=f"Q: {question}\nA: {answer}")
    
def main():
    root = tk.Tk()
    app = VQAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
