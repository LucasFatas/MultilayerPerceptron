import numpy as np

from Perceptron import Perceptron


# Initially, we can use the squared error function
# it could be changed later on
def calculate_error(prediction, actual):
    return np.square(np.subtract(actual, prediction))/2


def sigmoid_derivative(output):
    return output * (1 - output)


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
                layer.append(Perceptron(np.random.randn(size[i - 1]), 0))
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
            print(layer_results)
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
        next_layer_derivatives = []
        actual = vectorize(target, len(self.network[-1]))

        for l_index, layer in reversed(list(enumerate(self.network))):

            layer_derivatives = np.empty(len(self.network[l_index]))

            if l_index != 0:
                previous_layer_results = list(map(lambda x: x.output, self.network[l_index-1]))
            else:
                previous_layer_results = features

            for p_index, p in enumerate(layer):

                if l_index == (len(self.network)-1):
                    derivative = sigmoid_derivative(p.z) * (p.output - actual[p_index])
                else:
                    next_layer_weights = list(map(lambda x: x.weights[l_index], self.network[l_index+1]))
                    derivative = sigmoid_derivative(p.z) * np.dot(next_layer_weights, next_layer_derivatives)
                p.weights = p.weights + alpha * derivative * np.array(previous_layer_results)
                p.bias = p.bias + alpha * derivative
                layer_derivatives[p_index] = derivative

            next_layer_derivatives = np.array(layer_derivatives)

    def train(self, features, classes, alpha):
        predictions = []
        sum_error = 0
        for i in range(150):
            print(i)
            pred = self.feedforward(features[i])
            predictions.append(pred)

            self.backpropagation(features[i], classes[i], alpha)

        return predictions
























