from NeuralNetwork import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 3, 4])


def train_network():
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = np.genfromtxt("data/targets.txt")

    neuralnetwork = NeuralNetwork([10, 8, 7])

    lweight = 0.01
    predictions = neuralnetwork.train(features, targets, lweight)
    file = open("Group_18_classes.txt", "w+")
    for i in range(len(predictions)):
        file.write(str(predictions[i]) + ", ")
    file.close()

train_network()