import numpy as np

class Perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.z = None
        self.output = None

    def activation_function(self, z):
        return 1/(1 + np.exp(-z))

    def calculate_output(self, input):
        self.z = np.dot(input, self.weights) + self.bias
        self.output = self.activation_function(np.dot(input, self.weights) + self.bias)
        return self.output

    def toString(self):
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))
