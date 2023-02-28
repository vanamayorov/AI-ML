import numpy as np


class AdditionalNN:
    def __init__(self):
        self.weights = np.random.random((1, 3))
        self.prediction_output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        self.prediction_output = np.dot(self.weights, np.array(inputs))
        return self.sigmoid(self.prediction_output)

    def train(self, inputs, expected_predict):
        outputs = self.sigmoid(np.dot(self.weights, np.array(inputs)))
        error = expected_predict - outputs
        weights_delta = error * self.sigmoid(outputs) * (1 - self.sigmoid(outputs))
        self.weights += np.dot(weights_delta, inputs.reshape(1, len(inputs)))

    def start_training(self, epochs, data):
        print('AdditionalNN results on training data: ')
        for e in range(epochs):
            if e == epochs - 1:
                print(f"***EPOCH №{e+1}***")
            for input_stat, correct_predict in data:
                self.train(np.array(input_stat), correct_predict)
                if e == epochs - 1:
                    res = self.predict(input_stat)[0]
                    print("For Input: {}, Expected: {}, Result: {}".format(
                        str(input_stat),
                        str(correct_predict),
                        str(int(res > .5))
                    ))
