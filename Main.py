from NeuralNetwork import NeuralNetwork
import numpy as np


def outputAccuracyScore(neuralnetwork, features, targets):

    totalCorrect = 0
    for j in range(len(targets)):
        prediction = neuralnetwork.feedforward(features[j])
        if prediction == targets[j]:
            totalCorrect += + 1
    return totalCorrect / len(targets)

def train_network():
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = np.genfromtxt("data/targets.txt")

    neuralnetwork = NeuralNetwork([10, 7])

    lweight = 0.1
    predictions = neuralnetwork.train(features, targets, lweight)
    print(outputAccuracyScore(neuralnetwork, features, targets))
    #file = open("Group_18_classes.txt", "w+")
    #for i in range(len(predictions)):
        #file.write(str(predictions[i]) + ", ")
    #file.close()

train_network()