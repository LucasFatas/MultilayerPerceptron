from NeuralNetwork import NeuralNetwork
from sklearn.metrics import confusion_matrix
import numpy as np
import math
# these 2 imports are only used for the confusion matrix, not the neural network
import matplotlib.pyplot as plt
import seaborn as sns

# The network is split into files: Perceptron.py and NeuralNetwork.py. The Perceptron.py has values related to the
# neurons stored, contains the activation function and calculates the outputs for a single perceptron
# The Neural Network contains the different layers of the network and has three main functions: initialise the network,
# feedforward and backpropagation. In main we have ...
#

features_train = [(0, 0), (1, 0), (0, 1), (1, 1)]
features_test = [(0, 1), (1, 0), (1, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 0), (1, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)]


def or_function():
    targets_train = [0, 1, 1, 1]
    targets_test = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
    nn1 = NeuralNetwork([2, 1, 2])
    nn1.train(features_train, targets_train, 0.1, 500)
    print(outputAccuracyScore(nn1, features_test, targets_test))


def and_function():
    targets_train = [0, 0, 0, 1]
    targets_test = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    nn2 = NeuralNetwork([2, 1, 2])
    nn2.train(features_train, targets_train, 0.1, 500)
    print(outputAccuracyScore(nn2, features_test, targets_test))

def xor_function():
    targets_train = [0, 1, 1, 0]
    targets_test = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    nn3 = NeuralNetwork([2, 1, 2])
    nn3.train(features_train, targets_train, 0.1, 500)
    print(outputAccuracyScore(nn3, features_test, targets_test))


def outputAccuracyScore(neuralnetwork, features, targets):
    totalCorrect = 0
    predictions = []
    for j in range(len(targets)):
        prediction = neuralnetwork.feedforward(features[j])
        if prediction == targets[j]:
            totalCorrect += + 1
        predictions.append(prediction)
    # plot the confusion matrix of the test set
    #plot_cf(predictions, targets)
    return totalCorrect / len(targets)


def crossvalidation(neuralnetwork, training, target, k, alpha):
    sample_size = len(training)
    validation_size = math.floor(sample_size / k)  # the k in k-crossvalidation
    result = 0
    for x in range(10):  # do the cross-validation step 10 times and take the average since the initialization is random
        error = 0
        validation_end = 0
        for i in range(1, k+1):  # this is the cross-validation step
            nn = neuralnetwork
            validation_begin = validation_end
            validation_end = validation_size * i
            training_set = np.delete(training, range(validation_begin, validation_end), 0)
            validation_set = training[validation_begin:validation_end]
            training_target = np.delete(target, range(validation_begin, validation_end), 0)
            validation_target = target[validation_begin:validation_end]
            nn.train(training_set, training_target, alpha, 3)
            error += outputAccuracyScore(nn, validation_set, validation_target) / k
            print(error)
        result = result + error
    return result / 10


def find_optimal_neuron_amount(training, target, k, alpha):
    result = []
    index = 0
    for neurons in [7,8,10,15,25,30]:  # The amount of neurons is hardcoded for now
        nn = NeuralNetwork([10, neurons, 7])
        value = crossvalidation(nn, training, target, k, alpha)
        result.append(value) #This wont assign it for some reason so I just printed the values to do it by hand
        index = index + 1
        print("\n",result)
    return result


# this method creates a new neural network and then calls the train method in the NeuralNetwork to train it.
def train_network(data, data_targets, hidden_neurons, lweight, epochs):
    # create new neural network, hardcoded 1 hidden layer here.
    neuralnetwork = NeuralNetwork([10, hidden_neurons, 7])

    # split the data into a test set and a training set
    training = data[:(len(data) // 10) * 8]
    test = data[(len(data) // 10) * 8:]
    training_targets = data_targets[:(len(data_targets) // 10) * 8]
    test_targets = data_targets[(len(data_targets) // 10) * 8:]

    neuralnetwork.train(training, training_targets, lweight, epochs)
    # print the accuracy score and plot the confusion matrix
    print(outputAccuracyScore(neuralnetwork, test, test_targets))
    return neuralnetwork

#or_function()
#and_function()
#xor_function()


# takes a trained network and then writes the predictions of the unknown data set to the Group_18_classes.txt file
def unknowns(network):
    unknown = np.genfromtxt("data/unknown.txt", delimiter=",")
    predictions = []
    for i in range(len(unknown)):
        predictions.append(network.feedforward(unknown[i]))

    file = open("Group_18_classes.txt", "w+")
    for i in range(len(predictions)):
        if i < len(predictions) - 1:
            file.write(str(predictions[i]) + ", ")
        else:
            file.write(str(predictions[i]))
    file.close()

# method to plot the confusion matrix of the test set
def plot_cf(predictions, actual):
    cf = confusion_matrix(predictions, actual)
    hp = sns.heatmap(cf, annot=True, cmap='Blues', fmt='g')

    hp.set_title('Confusion Matrix Test Set Grocery Robot\n');
    hp.set_xlabel('Predicted Category')
    hp.set_ylabel('Actual Category ');
    hp.xaxis.set_ticklabels(['1', '2', '3', '4', '5', '6', '7'])
    hp.yaxis.set_ticklabels(['1', '2', '3', '4', '5', '6', '7'])

    plt.show()


# main method
if __name__ == "__main__":
    # fetch data from files
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = np.genfromtxt("data/targets.txt")
    # learning weight, amount of hidden neurons and amount of epochs are instantiated here
    alpha = 0.1
    neurons = 15
    epoch = 5
    # trains a new neural network and then feeds unknown.txt as input to it
    unknowns(train_network(features, targets, neurons, alpha, epoch))

    # how it would look to call the crossvalidation method
    # crossvalidation(NeuralNetwork([10, neurons, 7]), features, targets, 10, alpha)
    # how it would look to call the find_optimal_neuron_amount method
    # find_optimal_neuron_amount(features, targets, 10, alpha)
