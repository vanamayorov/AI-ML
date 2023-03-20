from tkinter import *
from PIL import EpsImagePlugin
from PIL import Image
from random import randint
import os

EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs10.00.0\bin\gswin64c'


class Paint(Frame):
    def __init__(self, parent, nn):
        Frame.__init__(self, parent)

        self.parent = parent
        self.color = "black"
        self.brush_size = 4
        self.last_save_file = None
        self.nn = nn
        self.prediction = None
        self.setUI()

    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

    def setUI(self):
        self.parent.title("Lab2")
        self.pack(fill=BOTH, expand=1)

        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)

        self.canv = Canvas(self, bg="white", height=200, width=200)
        self.canv.grid(row=2, column=0, columnspan=7)
        self.canv.bind("<B1-Motion>", self.draw)

        clear_btn = Button(self, text="Clear all", width=10,
                           command=lambda: self.canv.delete("all"))
        clear_btn.grid(row=0, column=1, sticky=W)

        recognize_btn = Button(self, text="Recognize symbol", width=40, command=self.save_paint)
        recognize_btn.grid(row=0, column=4, sticky=W)

    def refresh_letter_prediction(self, prediction):
        result = Label(self, text=f'Prediction: {prediction}')
        result.grid(row=1, column=1)

    def save_paint(self):
        filename = './drawings/my_drawing' + str(randint(0, 1000))
        self.last_save_file = filename + '.png'
        self.canv.postscript(file=filename + '.eps')

        img = Image.open(filename + '.eps')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img.save(self.last_save_file, 'png')
        os.remove(filename + '.eps')
        self.predict_img(self.last_save_file)

    def predict_img(self, image):
        prediction = self.nn.predict(image)
        self.refresh_letter_prediction(prediction)
