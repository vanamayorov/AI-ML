import numpy as np


class XorLogic:
    def __init__(self, input1, input2):
        self.inputs = np.array([input1, input2])
        self.input_layer_weights = np.array([[1, -1], [-1, 1]])
        self.second_layer_weights = np.array([1, 1])
        self.second_layer_activation_func_res = [self.activation_func(np.dot(self.inputs, self.input_layer_weights[i]))
                                                 for i in range(len(self.input_layer_weights))]

    def get_result(self):
        return self.activation_func(np.dot(self.second_layer_activation_func_res, self.second_layer_weights))

    def activation_func(self, x):
        if x >= 0.5:
            return 1
        return 0
