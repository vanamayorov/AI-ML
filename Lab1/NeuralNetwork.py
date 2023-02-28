import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.input_layer_weights = np.random.normal(0.0, 1, (3, 3))
        self.hidden_layer_weights = np.random.normal(0.0, 1, (1, 3))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])
        self.prediction_output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_1 = np.dot(self.input_layer_weights, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.hidden_layer_weights, outputs_1)
        self.prediction_output = inputs_2[0]
        outputs_2 = self.sigmoid_mapper(inputs_2)
        return outputs_2

    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.input_layer_weights, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.hidden_layer_weights, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.hidden_layer_weights -= np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.hidden_layer_weights
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.input_layer_weights -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1) * self.learning_rate

    def start_training(self, epochs, data):
        print('NeuralNetwork results on training data: ')
        for e in range(epochs):
            if e == epochs - 1:
                print(f"***EPOCH №{e+1}***")
            for input_stat, correct_predict in data:
                self.train(np.array(input_stat), correct_predict)
                if e == epochs - 1:
                    self.predict(input_stat)
                    print("Expected: {}, Result: {}, Difference: {}".format(
                        str(correct_predict),
                        str(self.prediction_output),
                        str(abs(correct_predict - self.prediction_output))
                    ))
