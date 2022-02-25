import numpy as np

from Perceptron import Perceptron


# Initially, we can use the squared error function
# it could be changed later on
def calculate_error(prediction, actual):
    return np.square(np.subtract(actual, prediction))


def sigmoid_derivative(z):
    return z * (1 - z)


def vectorize(target, size):
    vector = np.zeros(size)
    vector[int(target)-1] = 1
    return vector


class NeuralNetwork:

    # size should be an array, first value is number of inputs, last value is number of outputs and numbers in between
    # are the number of perceptrons in hidden layers
    def __init__(self, size):

        self.size = size

        self.network = []

        # creates every layer
        for i in range(1, len(size)):
            layer = []
            # creates every perceptron in layer
            for n in range(size[i]):
                # Random number for weights from 0 to 1 for now and bias of 0
                # WILL NEED TO BE CHANGED
                layer.append(Perceptron(np.random.randn(size[i - 1]), 0.01))
            self.network.append(layer)

    def toString(self):
        i = 0
        for layer in self.network:
            i += 1
            print("Layer " + str(i) + ": ")
            for perceptron in layer:
                perceptron.toString()

    def feedforward(self, inputs):

        layer_inputs = inputs
        for layer in self.network:
            # results of current layer
            layer_results = []
            for perceptron in layer:
                layer_results.append(perceptron.calculate_output(layer_inputs))
            # normalizes outputs
            layer_results = layer_results / np.sum(layer_results)
            for index, perceptron in enumerate(layer):
                perceptron.output = layer_results[index]
            # layer_inputs gets replaced by results of this layer so that next layer can use them as inputs
            layer_inputs = layer_results

            # Should maybe store results of each layer in perceptron since will be needed for backpropagation later

        # layer_inputs now how was results of the last layer of the network which is the predictions
        # the largest value should correspond to the index of the most likely output
        prediction = list(layer_inputs).index(max(layer_inputs)) + 1

        return prediction

    def backpropagation(self, features, target, alpha):

        self.feedforward(features)
        actual = vectorize(target, len(self.network[-1]))

        for l_index, layer in reversed(list(enumerate(self.network))):

            if l_index != 0:
                previous_layer_results = list(map(lambda x: x.output, self.network[l_index-1]))
            else:
                previous_layer_results = features

            for p_index, p in enumerate(layer):

                if l_index == (len(self.network)-1):
                    p.derivative = sigmoid_derivative(p.output) * (actual[p_index] - p.output)
                else:
                    next_layer_weights_delta_sum = sum(list(map(lambda x: x.weights[p_index] * x.derivative,
                                                            self.network[l_index+1])))
                    p.derivative = sigmoid_derivative(p.output) * next_layer_weights_delta_sum

                p.weights = p.weights + alpha * p.derivative * np.array(previous_layer_results)
                p.bias = p.bias + alpha * p.derivative

    def train(self, features, classes, alpha):
        for i in range(len(features)):
            self.backpropagation(features[i], classes[i], alpha)

