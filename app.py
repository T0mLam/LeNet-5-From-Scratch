import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import ImageGrab, ImageOps
import cv2
import matplotlib.pyplot as plt

from modules.utils import load_model


class DrawingCanvas(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(height=400, width=400, bg='white')
        self.bind('<B1-Motion>', self.draw)
        self.brush_size = 48.5

    def draw(self, event):
        x, y = event.x, event.y
        r = self.brush_size / 2
        self.create_oval((x - r, y - r, x + r, y + r), fill='black')
    
    def clear(self):
        self.delete('all')

    def get_canvas(self):
        return ImageGrab.grab(bbox=(
            self.winfo_rootx(),
            self.winfo_rooty(),
            self.winfo_rootx() + self.winfo_width(),
            self.winfo_rooty() + self.winfo_height()
        ))


class PredictButton(ctk.CTkButton):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.configure(
            text="Predict", 
            height=40,
	        width=40,
            command=app.predict,
            corner_radius=30
        )
        

class ResetButton(ctk.CTkButton):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.canvas = app.canvas
        self.configure(
            text="Reset", 
            command=self.reset_canvas, 
            fg_color="red",
            hover_color='#8B0000',
            height=40,
	        width=40,
            corner_radius=30
        )

    def reset_canvas(self):
        self.canvas.clear()

    
class ButtonFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure((0, 1), weight=1)

        self.predict_button = PredictButton(self, parent)
        self.predict_button.grid(row=0, column=0, padx=10, sticky='e')

        self.reset_button = ResetButton(self, parent)
        self.reset_button.grid(row=0, column=1, padx=10, sticky='w')


class DigitRecognitionApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title('Digit Recognition')
        self.geometry('600x800')
        self.resizable(False, False)

        self.canvas = DrawingCanvas(self)
        self.canvas.pack(pady=40)

        self.button_frame = ButtonFrame(self)
        self.button_frame.pack(pady=10, fill='x')

        self.model = load_model(model)

    def predict(self):
        img = self.canvas.get_canvas()
        img = img.resize((32, 32)).convert('L')
        img = ImageOps.invert(img)
        #img.save('pred.png')
        img_array = np.array(img) / 255

        self.model.eval()
        pred = self.model(img_array.reshape(1, 1, 32, 32)).argmax(axis=1)
        print(pred)
        
        
if __name__ == '__main__':
    app = DigitRecognitionApp(model='models/model_10ep.pkl')
    app.mainloop()