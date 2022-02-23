import numpy as np

class Perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activation_function(self, x):
        result = 1 / (1 + pow(np.exp, -x))
        return result

    def calculate_output(self, input):
        return self.activation_function(self, np.dot(input, self.weights) + self.bias)

    def toString(self):
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))
