from NeuralNetwork import NeuralNetwork


nn = NeuralNetwork([2, 3, 4])

def network():
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = np.genfromtxt("data/targets.txt")

    neuralnetwork = NeuralNetwork([10, 8, 7])
    for i in len(features):
        nn.backpropagation(features[i], targets[i], 0.1)
    nn.feedforward()
