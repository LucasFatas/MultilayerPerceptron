import numpy as np

class Perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # Incorrect NEEDS activation function
    def calculate_output(self, input):
        return np.dot(input, self.weights) + self.bias

    def toString(self):
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))
