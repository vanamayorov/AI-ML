import string
from extra_keras_datasets import emnist
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Flatten
import cv2
import os


class NeuralNetwork:
    def __init__(self):
        self.model = None
        self.x_train = None
        self.y_train = None
        self.letters_dict = dict(zip(range(1, 27), string.ascii_lowercase))
        self.init_nn()

    def init_nn(self):
        self.init_model()
        self.load_data()

    def init_model(self):
        if os.path.exists('letters.model'):
            self.model = keras.models.load_model('letters.model')
        else:
            model = keras.Sequential([
                Flatten(input_shape=(28, 28, 1)),
                Dense(128, activation='relu'),
                Dense(128, activation='relu'),
                Dense(27, activation='softmax')
            ])
            self.model = model

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = emnist.load_data(type='letters')
        self.x_train = x_train / 255
        self.y_train = y_train

    def train(self, epochs):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, batch_size=64, epochs=epochs)
        self.model.save('letters.model')

    def predict(self, filename):
        image = cv2.imread(filename)[:, :, 0]
        image = np.invert(np.array([image]))
        prediction = self.model.predict(image)
        print(prediction)
        return self.letters_dict[np.argmax(prediction)]
