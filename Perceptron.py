import numpy as np

from NeuralNetwork import NeuralNetwork

class Perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate_output(self, input):
        return NeuralNetwork.activation_function(self, np.dot(input, self.weights) + self.bias)

    def toString(self):
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))
