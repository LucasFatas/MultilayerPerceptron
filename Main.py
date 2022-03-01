from NeuralNetwork import NeuralNetwork
import numpy as np
import math

features = [(0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 0), (1, 0), (0, 0), (1, 1), (1, 1), (1, 1)]

def or_function():
    targets = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    nn = NeuralNetwork([2, 2, 2])
    nn.train(features, targets, 0.1)
    print(outputAccuracyScore(nn, features, targets))


def and_function():
    targets = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]
    nn = NeuralNetwork([2, 2, 2])
    nn.train(features, targets, 0.1)
    print(outputAccuracyScore(nn, features, targets))

def xor_function():
    targets = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    nn = NeuralNetwork([2, 2, 2])
    nn.train(features, targets, 0.1)
    print(outputAccuracyScore(nn, features, targets))


def outputAccuracyScore(neuralnetwork, features, targets):
    totalCorrect = 0
    for j in range(len(targets)):
        prediction = neuralnetwork.feedforward(features[j])
        if prediction == targets[j]:
            totalCorrect += + 1
    return totalCorrect / len(targets)


def crossvalidation(neuralnetwork ,training, target, k, alpha):
    sample_size = len(training)
    validation_size = math.floor(sample_size / k)  # the k in k-crossvalidation
    for x in range(10):  # do the cross-validation step 10 times and take the average since the initialization is random
        error = 0
        validation_end = 0
        for i in range(1, k+1):  # this is the cross-validation step
            validation_begin = validation_end
            validation_end = validation_size * i
            training_set = np.delete(training, range(validation_begin, validation_end), 0)
            validation_set = training[validation_begin:validation_end]
            training_target = np.delete(target, range(validation_begin, validation_end), 0)
            validation_target = target[validation_begin:validation_end]
            neuralnetwork.train(training_set, training_target, alpha)
            error += outputAccuracyScore(neuralnetwork, validation_set, validation_target) / k
    return error / 10


def find_optimal_neuron_amount(training, target):
    result = np.array(23)
    for neurons in range(7, 31):  # The amount of neurons is hardcoded for now
        error = 0
        nn = NeuralNetwork([10, neurons, 7])
        error += crossvalidation(nn, training, target)
        result[neurons - 7] = error
    return result


def train_network():
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = np.genfromtxt("data/targets.txt")

    neuralnetwork = NeuralNetwork([10, 8, 7])

    lweight = 0.1
    predictions = neuralnetwork.train(features, targets, lweight)
    print(outputAccuracyScore(neuralnetwork, features, targets))

    #print(crossvalidation(neuralnetwork, features, targets, 20, lweight))

    file = open("Group_18_classes.txt", "w+")
    for i in range(len(predictions)):
        if i < len(predictions) - 1:
            file.write(str(predictions[i]) + ", ")
        else:
            file.write(str(predictions[i]))
    file.close()

#or_function()
#and_function()
#xor_function()
train_network()
