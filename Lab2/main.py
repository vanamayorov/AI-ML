from tkinter import *
import os
from NeuralNetwork import NeuralNetwork
from Paint import Paint


def main():
    nn = NeuralNetwork()
    if not os.path.exists('letters.model'):
        nn.train(5)
    root = Tk()
    root.geometry("500x500")
    app = Paint(root, nn)
    root.mainloop()


if __name__ == '__main__':
    main()
