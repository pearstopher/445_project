# CS445 Term Project
# Christopher Juncker
#
# The end goal of this project is to create a neural network which can transform
# one type of sound to another.
#
#
# todo: create fake dataset (pairs of data)
# todo: modify network to have an equal number of input and output nodes
#

import os
import numpy as np
from matplotlib import pyplot as plt
from math import exp
import scipy.io.wavfile as wf


# "Set the learning rate to 0.1 and the momentum to 0.9.
ETA = 0.1
MOMENTUM = 0.9

# "Train your network for 50 epochs"
MAX_EPOCHS = 100

# "Experiment 1: Vary number of hidden units.
# "Do experiments with n = 20, 50, and 100.
# "(Remember to also include a bias unit with weights to every hidden and output node.)
N = 20


# class for loading and preprocessing data
# data is contained in a numpy array
class Data:
    def __init__(self):

        # 1. read in the audio files

        self.SINE_DIR = "audio/sine"  # the sine wave is the input
        self.SQUARE_DIR = "audio/square"  # the square wave is the ground truth

        files = os.listdir(self.SINE_DIR)
        # self.sine = [np.empty(0) * len(files)]
        # self.square = [np.empty(0) for _ in range(len(files))]  # same?
        self.sine = np.empty((0, 500))
        self.square = np.empty((0, 500))

        for i, file in enumerate(files):
            with open(os.path.join(self.SINE_DIR, file), 'r') as f:
                # self.sine[i] = wf.read(f)[:500]  # limit to first 500 samples for initial tests
                np.append(self.sine, wf.read(f)[0:500])

            with open(os.path.join(self.SQUARE_DIR, file), 'r') as f:
                # self.square[i] = wf.read(f)[:500]
                np.append(self.square, wf.read(f)[0:500])

        # 2. preprocess and augment the data
        self.preprocess()
        self.augment()

        # 3. split the data into a testing and training set
        self.training_data, self.training_truth, \
            self.testing_data, self.testing_truth = self.test_train_split()

    def test_train_split(self):
        # randomly shuffle the input and truth arrays (together)
        length = self.sine.shape[0]
        indices = np.arange(length)
        np.random.shuffle(indices)
        self.sine = self.sine[indices]
        self.square = self.square[indices]

        # perform an 80 / 20 split on the shuffled data
        split = int(length*0.8)
        training_data = self.sine[0:split]
        testing_data = self.sine[split:]
        training_truth = self.square[0:split]
        testing_truth = self.square[split:]

        return training_data, training_truth, testing_data, testing_truth

    # Preprocessing (not yet implemented)
    def preprocess(self):
        return

    # Augmentation (not yet implemented)
    def augment(self):
        return

    # return the testing dataset
    def test(self):
        return self.testing_data, self.testing_truth

    # return the training dataset
    def train(self):
        return self.training_data, self.training_truth


# very simple custom confusion matrix
class ConfusionMatrix:
    def __init__(self):
        self.matrix = np.zeros((10, 10))

    def insert(self, true, pred):
        self.matrix[true][pred] += 1


# "Your neural network will have 784 inputs, one hidden layer with
# "n hidden units (where n is a parameter of your program), and 10 output units.
class NeuralNetwork:
    def __init__(self, eta, momentum):
        print("Initializing neural network...")
        # set learning rate and momentum
        self.eta = eta
        self.momentum = momentum

        # explicitly set the size of these arrays (for matrix multiplication / my own sanity)
        # input layer:                              1 x 785
        # hidden layer weights:                     785 x (N + 1)
        # input layer (dot) hidden layer weights:   1 x (N + 1)
        #
        # hidden layer:                             1 x (N + 1)
        # output layer weights:                     (N + 1) x 10
        # hidden layer (dot) output layer weights:  1 x 10
        #
        #
        # "Choose small random initial weights, 𝑤! ∈ [−.05, .05]
        self.hidden_layer_weights = np.random.uniform(-0.05, 0.05, (785, N + 1))
        self.hidden_layer_weights_change = np.zeros((785, N + 1))  # save momentum
        self.hidden_layer = np.zeros((N+1))
        self.hidden_layer[0] = 1  # bias
        self.output_layer_weights = np.random.uniform(-0.05, 0.05, (N+1, 10))
        self.output_layer_weights_change = np.zeros((N+1, 10))  # save momentum
        self.output_layer = np.zeros(10)

    # "The activation function for each hidden and output unit is the sigmoid function
    # σ(z) = 1 / ( 1 + e^(-z) )
    @staticmethod
    def sigmoid(value):
        activation = 1 / (1 + exp(-value))
        return activation

    # "Compute the accuracy on the training and test sets for this initial set of weights,
    # "to include in your plot. (Call this “epoch 0”.)
    def compute_accuracy(self, data, freeze=False, matrix=None):
        num_correct = 0

        # for each item in the dataset
        for d, truth in zip(data[0], data[1]):

            #####################
            # FORWARD PROPAGATION
            #####################

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            self.hidden_layer = np.dot(d, self.hidden_layer_weights)
            self.hidden_layer = np.array([self.sigmoid(x) for x in self.hidden_layer])
            # print(self.hidden_layer)  # these are the right size as expected

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            self.output_layer = np.array([self.sigmoid(x) for x in self.output_layer])
            # print(self.output_layer)  # these are the right size as expected

            # (for report)
            # add our result to the confusion matrix
            if matrix:
                matrix.insert(int(np.argmax(self.output_layer)), int(truth), )  # x=pred, y=true

            # "If this is the correct prediction, don’t change the weights and
            # "go on to the next training example.
            if truth == np.argmax(self.output_layer):
                num_correct += 1

            ##################
            # BACK-PROPAGATION
            ##################

            elif not freeze:
                output_error = np.empty(10)
                hidden_error = np.empty(N + 1)

                # "For each output unit k, calculate error term δ_k
                # δ_k <- o_k (1 - o_k) (t_k - o_k)
                #
                # t = true value
                # I am guessing -1 is true value when it's wrong
                # and 1 is true value when its right
                for k, o_k in enumerate(self.output_layer):
                    # "Set the target value tk for output unit k to 0.9 if the input class is the kth class,
                    # "0.1 otherwise
                    t_k = 0.9 if truth == k else 0.1  # had truth == o_k instead of truth == the index

                    error = o_k * (1 - o_k) * (t_k - o_k)
                    output_error[k] = error

                # "for each hidden unit j, calculate error term δ_j (k = output units)
                # δ_j <- h_j (1 - h_j) (Σ_k ( w_kj * δ_k ) )
                for j, h_j in enumerate(self.hidden_layer):
                    # calculate sum
                    total = 0
                    # print(self.output_layer)
                    for k in range(len(self.output_layer)):
                        total += self.output_layer_weights[j][k] * output_error[k]

                    error = h_j * (1 - h_j) * total
                    hidden_error[j] = error  # oops was appending still

                # "Hidden to Output layer: For each weight w_kj
                # w_kj = w_kj + Δw_kj
                # Δw_kj = η * δ_k * h_j
                self.output_layer_weights_change = \
                    self.eta * (self.hidden_layer.reshape(N+1, 1) @ output_error.reshape(1, 10)) + \
                    self.momentum * self.output_layer_weights_change
                self.output_layer_weights += self.output_layer_weights_change

                # "Input to Hidden layer: For each weight w_ji
                # w_ji = w_ji + Δw_ji
                # Δw_ji = η * δ_j * x_i
                self.hidden_layer_weights_change = \
                    self.eta * (d.reshape(785, 1) @ hidden_error.reshape(1, N+1)) + \
                    self.momentum * self.hidden_layer_weights_change
                self.hidden_layer_weights += self.hidden_layer_weights_change

        # return accuracy
        return num_correct / len(data[0])

    def run(self, data, matrix, epochs):
        train_accuracy = []
        test_accuracy = []

        print("Epoch 0: ", end="")
        train_accuracy.append(self.compute_accuracy(data.train(), True))
        test_accuracy.append(self.compute_accuracy(data.train(), True))
        print("Training Set:\tAccuracy:", "{:0.5f}".format(train_accuracy[0]), end="\t")
        print("Testing Set:\tAccuracy:", "{:0.5f}".format(test_accuracy[0]))

        for i in range(epochs):
            print("Epoch " + str(i + 1) + ": ", end="")
            train_accuracy.append(self.compute_accuracy(data.train()))
            test_accuracy.append(self.compute_accuracy(data.train(), True))
            print("Training Set:\tAccuracy:", "{:0.5f}".format(train_accuracy[i + 1]), end="\t")
            print("Testing Set:\tAccuracy:", "{:0.5f}".format(test_accuracy[i + 1]))

        # "Confusion matrix on the test set, after training has been completed.
        self.compute_accuracy(data.train(), True, matrix)

        return train_accuracy, test_accuracy


def main():

    d = Data()
    p = NeuralNetwork(ETA, MOMENTUM)
    c = ConfusionMatrix()

    results = p.run(d, c, MAX_EPOCHS)

    # plot the training / testing accuracy
    plt.plot(list(range(MAX_EPOCHS + 1)), results[0])
    plt.plot(list(range(MAX_EPOCHS + 1)), results[1])
    plt.xlim([0, MAX_EPOCHS])
    plt.ylim([0, 1])
    plt.show()

    # plot the confusion matrix
    for i in range(10):
        plt.plot([-0.5, 9.5], [i+0.5, i+0.5], i, color='xkcd:chocolate', linewidth=1)  # nice colors
        plt.plot([i+0.5, i+0.5], [-0.5, 9.5], i, color='xkcd:chocolate', linewidth=1)
        for j in range(10):
            plt.scatter(i, j, s=(c.matrix[i][j] / 3), c="xkcd:fuchsia", marker="s")  # or chartreuse
            plt.annotate(int(c.matrix[i][j]), (i, j))
    plt.xlim([-0.5, 9.5])
    plt.ylim([-0.5, 9.5])
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
