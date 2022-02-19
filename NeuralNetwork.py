import numpy as np


class NeuralNetwork:

    def __init__(self, size):

        self.size = size

        self.network = []

        for i in range(len(size)):
            layer = []
            for n in range(len(size[i])):
                # Random number for weights from 0 to 1 for now and bias of 0
                # WILL NEED TO BE CHANGED
                layer.append(Perceptron(np.random.radn(size[i, n], 1), 0))
            self.network.append(layer)

    def feedforward(self, inputs):

        layer_inputs = inputs
        for layer in self.network:
            # results of current layer
            layer_results = []
            for perceptron in layer:
                layer_results.append(perceptron.calculate(layer_inputs))

            # layer_inputs gets replaced by results of this layer so that next layer can use them as inputs
            layer_inputs = layer_results

        # layer_inputs now how was results of the last layer of the network which is the predictions
        # largest value should correspond to the index of the most likely output
        prediction = layer_inputs.index(max(layer_inputs))

        return prediction

    def backpropagation(self):
        pass

    def activation_function(self):
        pass
