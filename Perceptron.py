

class Perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate_output(self, input):
        return input * self.weight + bias

