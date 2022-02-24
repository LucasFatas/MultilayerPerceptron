from NeuralNetwork import NeuralNetwork


nn = NeuralNetwork([2, 3, 4])

nn.toString()

print(nn.feedforward([1, -1]))

print(nn.calculate_error([0, 1], [2, 2]))
