import numpy as np

from NeuralNetwork import NeuralNetwork


class Optimization:

    def crossvalidation(self, neuralnetwork ,training, target):
        sample_size = len(training)
        validation_size = sample_size / 20  # the k in k-crossvalidation is hardcoded to 20 here
        validation_end = 0
        for x in range(10):  # do the cross-validation step 10 times and take the average since the initialization is random
            error1 = 0
            for i in range(1, 21):  # this is the cross-validation step
                validation_begin = validation_end
                validation_end = validation_size * i
                validation_set = np.take(training, range(validation_begin, validation_end))
                training_set = np.delete(training, range(validation_begin, validation_end), None)
                validation_target = np.take(target, range(validation_begin, validation_end))
                training_target = np.delete(target, range(validation_begin, validation_end), None)
                error1 += neuralnetwork.train(training_set, training_target, validation_set,
                                   validation_target) / 20  # for now I assume train returns error (might be changed later)
        return error1 / 10

    def find_optimal_neuron_amount(self, training, target):
        result = np.array(23)
        for neurons in range(7, 31):  # The amount of neurons is hardcoded for now
            nn = NeuralNetwork([10, neurons, 7])
            error2 += crossvalidation(nn, training, target) / 20
            result[neurons - 7] = error2 / 10
        return result
