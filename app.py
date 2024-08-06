import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import ImageGrab, ImageOps
import cv2
import matplotlib.pyplot as plt

from modules.utils import load_model, preprocessing


class DrawingCanvas(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.configure(height=400, width=400, bg='black')
        self.bind('<B1-Motion>', self.draw)
        self.brush_size = 48.5

    def draw(self, event):
        x, y = event.x, event.y
        r = self.brush_size / 2
        self.create_oval((x - r, y - r, x + r, y + r), fill='white', outline='white')
    
    def clear(self):
        self.delete('all')

    def get_canvas(self):
        return ImageGrab.grab(bbox=(
            self.winfo_rootx() + 3,
            self.winfo_rooty() + 3,
            self.winfo_rootx() + self.winfo_width() - 3,
            self.winfo_rooty() + self.winfo_height() - 3
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
            command=app.reset, 
            fg_color="red",
            hover_color='#8B0000',
            height=40,
	        width=40,
            corner_radius=30
        )

    
class ButtonFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure((0, 1), weight=1)

        self.predict_button = PredictButton(self, parent)
        self.predict_button.grid(row=0, column=0, padx=10, sticky='e')

        self.reset_button = ResetButton(self, parent)
        self.reset_button.grid(row=0, column=1, padx=10, sticky='w')


class PredictionLabel(ctk.CTkLabel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.configure(
            text='8',
            text_color='white',
            fg_color='black',
            width=120,
            height=120,
            font=('Bold', 80)
        )


class ConfidenceLabel(ctk.CTkLabel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.configure(
            text='Confidence:',
            font=('Helvetica', 18, 'bold'),
            text_color='#636363'
        )


class ConfidenceBar(ctk.CTkProgressBar):
    def __init__(self, parent, app):
        super().__init__(parent)
    
    def update_progess(self, prob, current_value=0):
        max_value = prob * 100
        if current_value <= max_value:
            self.set(current_value / 100)
            current_value += 1 
            self.after(5, self.update_progess, prob, current_value)


class PredictionFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(
            corner_radius=50,  
            fg_color="#e0e0e0"  
        )
        self.rowconfigure((0, 1), weight=1)
        self.columnconfigure((0, 1), weight=1)

        self.pred_label = PredictionLabel(self, parent)
        self.pred_label.grid(row=0, column=0, rowspan=2, padx=40, pady=20)

        self.conf_label = ConfidenceLabel(self, parent)
        self.conf_label.grid(row=0, column=1, padx=(0, 40), sticky='ew')

        self.conf_bar = ConfidenceBar(self, parent)
        self.conf_bar.grid(row=1, column=1, padx=(0, 40), sticky='new')

    def update_prediction_labels(self, pred, prob):
        self.pred_label.configure(text=pred, text_color='white')
        self.conf_label.configure(text=f'Confidence: {int(prob * 100)}%')
        self.conf_bar.update_progess(prob)
        self.conf_bar.configure(progress_color=self.label_color(prob))

    def label_color(self, prob):
        if prob > 0.85:
            return 'green'
        elif prob > 0.5:
            return '#FFBF00'
        else:
            return 'red'
        
    def update_empty_prediction_labels(self):
        self.pred_label.configure(text_color='black')
        self.conf_label.configure(text=f'Confidence: -')
        self.conf_bar.update_progess(0)


class DigitRecognitionApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title('Digit Recognition App')
        self.geometry('600x800')
        self.resizable(False, False)

        self.canvas = DrawingCanvas(self)
        self.canvas.pack(pady=20)

        self.button_frame = ButtonFrame(self)
        self.button_frame.pack(pady=10, fill='x')

        self.pred_frame = PredictionFrame(self)

        self.model = load_model(model)

    def predict(self):
        self.pred_frame.pack(pady=40)
        
        img = self.canvas.get_canvas()
        img_arr = preprocessing(img)

        if isinstance(img_arr, int):
            self.pred_frame.update_empty_prediction_labels()
            return 

        self.model.eval()
        res = self.model(img_arr.reshape(1, 1, 32, 32))
        pred = res.argmax(axis=1)[0]
        prob = round(res[0, pred], 2)

        print(pred)
        self.pred_frame.update_prediction_labels(pred, prob)
        
    def reset(self):
        self.canvas.clear()
        
        
if __name__ == '__main__':
    app = DigitRecognitionApp(model='models/model_10ep.pkl')
    app.mainloop()