import numpy as np


class LogicNeuron:
    def __init__(self, input1, input2=None):
        self.inputs = np.array([input1, input2]) if input2 is not None else np.array([input1])
        self.weights = np.array([1, 1]) if input2 is not None else np.array([1])

    def activation_func(self, x, value):
        if x >= value:
            return 1
        else:
            return 0

    def get_result(self, act_func_val):
        return self.activation_func(np.dot(self.inputs, self.weights), act_func_val)
