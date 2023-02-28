import numpy as np

from NeuralNetwork import NeuralNetwork
from AdditionalNN import AdditionalNN
from LogicNeuron import LogicNeuron
from XorLogic import XorLogic

if __name__ == '__main__':
    and_logic_neuron = LogicNeuron(1, 1)
    print("x1 = 1, x2 = 1 => x1 AND x2 = " + str(and_logic_neuron.get_result(1.5)))
    or_logic_neuron = LogicNeuron(0, 1)
    print("x1 = 0, x2 = 1 => x1 OR x2 = " + str(or_logic_neuron.get_result(0.5)))
    not_logic_neuron = LogicNeuron(0)
    print("x = 0, NOT x = " + str(not_logic_neuron.get_result(-1)))
    xor_neuron = XorLogic(1, 1)
    print("x1 = 1, x2 = 1, x1 xor x2 = " + str(xor_neuron.get_result()))

    values = [
        2.82, 3.48, 0.60, 4.76,
        1.51, 5.51, 1.48, 5.19,
        0.48, 5.22, 0.21, 4.19,
        0.07, 4.63, 0.49
    ]

    transformed_data_to_train = [([values[i], values[i + 1], values[i + 2]], values[i + 3]) for i in range(10)]
    transformed_data_to_predict = [([values[i], values[i + 1], values[i + 2]], values[i + 3]) for i in range(10, 12)]
    epochs = 5000
    learning_rate = 0.0005

    network = NeuralNetwork(learning_rate)
    network.start_training(epochs, transformed_data_to_train)

    print('NeuralNetwork results on test data: ')
    for input_stat, correct_predict in transformed_data_to_predict:
        network.predict(np.array(input_stat).T)
        print("Expected: {}, Result: {}, Difference: {}".format(
            str(correct_predict),
            str(network.prediction_output),
            str(abs(correct_predict - network.prediction_output))
        ))

    values = [
        ([0, 0, 0], 1),
        ([0, 1, 0], 1),
        ([1, 0, 0], 0),
        ([1, 1, 1], 1),
    ]

    network_additional = AdditionalNN()
    network_additional.start_training(epochs, values)
    print('AdditionalNN results: ')
    for input_stat, correct_predict in values:
        prediction = network_additional.predict(input_stat)[0]
        print("For Input: {}, Expected: {}, Result: {}, Prediction: {}".format(
            str(input_stat),
            str(correct_predict),
            str(int(prediction >= .5)),
            str(prediction)
        ))
