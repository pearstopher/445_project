# CS445 Term Project
# Christopher Juncker
#
# The end goal of this project is to create a neural network which can transform
# one type of sound to another.
#
# This file contains a neural network which can be trained on datasets from the
# /audio/datasets directory.
#

###########
# INCLUDES
###########

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wf

############
# CONSTANTS
############

# each item in the dataset will have a size of 500 samples
SAMPLES = 500

# during data augmentation, the program will step through each file in the dataset, creating multiple data entries.
# OFFSET is the step size
# OFFSET_LOOPS is the number of steps (and therefore the number of entries which will be created for each file)
OFFSET = 200
OFFSET_LOOPS = 100
# make sure OFFSET * OFFSET_LOOPS does not exceed the total number of samples in each input file.

# If desired, limit the number of input files. If set to 0, the whole dataset will be loaded.
MAX_FILES = 0


# Set the learning rate (ETA) and the momentum value.
ETA = 0.1
MOMENTUM = 0.0

# Set the number of epochs to run the program
MAX_EPOCHS = 50

# Set the number of hidden units (not counting the bias unit).
N = 400


###################
# LOADING THE DATA
###################

# class for loading, preprocessing, and augmenting the data
class Data:
    def __init__(self):

        # 1. read in the audio files
        self.INPUT_DIR = "audio/datasets/" + sys.argv[1] + "/"  # the square wave is the input
        self.TRUTH_DIR = "audio/datasets/" + sys.argv[2] + "/"  # the sine wave is the ground truth

        # 2. randomly order the files
        files = os.listdir(self.INPUT_DIR)
        np.random.shuffle(files)

        # 3. load the correct number of files
        num_files = len(files) if MAX_FILES == 0 else MAX_FILES
        self.input = np.empty((num_files * OFFSET_LOOPS, SAMPLES))
        self.truth = np.empty((num_files * OFFSET_LOOPS, SAMPLES))
        for i, file in enumerate(files):
            if MAX_FILES != 0 and i >= MAX_FILES:
                break

            # 4. for each input file, loop through and create a series of data items
            with open(os.path.join(self.INPUT_DIR, file), 'r') as f:
                _, samples = wf.read(f.name)
                for j in range(OFFSET_LOOPS):
                    self.input[i * OFFSET_LOOPS + j] = samples[0 + j * OFFSET:SAMPLES + j * OFFSET].reshape(1, SAMPLES)

            # 5. for each truth file, loop and create a series of matching data items
            with open(os.path.join(self.TRUTH_DIR, file), 'r') as f:
                _, samples = wf.read(f.name)
                for j in range(OFFSET_LOOPS):
                    self.truth[i * OFFSET_LOOPS + j] = samples[0 + j * OFFSET:SAMPLES + j * OFFSET].reshape(1, SAMPLES)

        # 6. preprocess and augment the data
        self.preprocess()
        self.augment()

        # 7. split the finished data into a testing and training set
        self.training_data, self.training_truth, \
            self.testing_data, self.testing_truth = self.test_train_split()

    # split the data into testing and training set
    def test_train_split(self):
        # randomly shuffle the input and truth arrays (together)
        length = self.input.shape[0]
        indices = np.arange(length)
        np.random.shuffle(indices)
        self.input = self.input[indices]
        self.truth = self.truth[indices]

        # perform an 80 / 20 split on the shuffled data
        split = int(length*0.8)
        training_data = self.input[0:split, :]
        testing_data = self.input[split:, :]
        training_truth = self.truth[0:split, :]
        testing_truth = self.truth[split:, :]

        return training_data, training_truth, testing_data, testing_truth

    # preprocess the data
    def preprocess(self):
        # normalize the data to be in the range (0, 1)
        self.input = (self.input + 2 ** 15) / 2 ** 16
        self.truth = (self.truth + 2 ** 15) / 2 ** 16
        # (this matches the range of values supported by the activation function)
        return

    # augment the data
    def augment(self):
        # currently all of the data augmentation is done as the data is loaded
        # however there are plenty of ways I could augment further if necessary
        return

    # return the testing dataset
    def test(self):
        return self.testing_data, self.testing_truth

    # return the training dataset
    def train(self):
        return self.training_data, self.training_truth


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
        # output layer weights:                     (N + 1) x 785
        # hidden layer (dot) output layer weights:  1 x 785
        self.input_len = SAMPLES
        self.output_len = SAMPLES
        #
        #
        # "Choose small random initial weights, 𝑤! ∈ [−.05, .05]
        self.hidden_layer_weights = np.random.uniform(-0.05, 0.05, (self.input_len, N + 1))
        self.hidden_layer_weights_change = np.zeros((self.input_len, N + 1))  # save momentum
        self.hidden_layer = np.zeros((N+1))
        self.hidden_layer[0] = 1  # bias
        self.output_layer_weights = np.random.uniform(-0.05, 0.05, (N+1, self.output_len))
        self.output_layer_weights_change = np.zeros((N + 1, self.output_len))  # save momentum
        self.output_layer = np.zeros(self.output_len)

    # "The activation function for each hidden and output unit is the sigmoid function
    # σ(z) = 1 / ( 1 + e^(-z) )
    @staticmethod
    def sigmoid(array):
        array = 1 / (1 + np.e ** (- array))
        return array
    # @staticmethod
    # def sigmoid(value):
    #     activation = 1 / (1 + exp(-value))
    #     return activation

    # trying new activation functions
    # @staticmethod
    # def softplus(value):
    #     # activation = np.log(1 + np.e**value)  # overflow baby
    #     activation = np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)
    #    return activation

    # "Compute the accuracy on the training and test sets for this initial set of weights,
    # "to include in your plot. (Call this “epoch 0”.)
    #
    # data[0] = input data
    # data[1] = truth data
    def compute_accuracy(self, data, epoch, freeze=False):
        avg = 0

        # randomly shuffle the input and truth arrays (together)
        data = np.copy(data)

        d0 = data[0]
        d1 = data[1]
        if not freeze:
            length = d1.shape[0]
            indices = np.arange(length)
            np.random.shuffle(indices)
            d0 = d0[indices, ]
            d1 = d1[indices, ]

        # for each item in the dataset
        # for d, truth in zip(data[0], data[1]):
        for count, (d, truth) in enumerate(zip(d0, d1)):

            #####################
            # FORWARD PROPAGATION
            #####################

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            self.hidden_layer = np.dot(d, self.hidden_layer_weights)
            # self.hidden_layer = np.array([self.sigmoid(x) for x in self.hidden_layer])
            self.hidden_layer = self.sigmoid(self.hidden_layer)

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            # self.output_layer = np.array([self.sigmoid(x) for x in self.output_layer])
            self.output_layer = self.sigmoid(self.output_layer)

            ##################
            # BACK-PROPAGATION
            ##################

            if not freeze:
                output_error = np.empty(self.output_len)
                hidden_error = np.empty(N + 1)

                # "For each output unit k, calculate error term δ_k
                # δ_k <- o_k (1 - o_k) (t_k - o_k)
                #
                # t = true value
                for k, o_k in enumerate(self.output_layer):
                    # "Set the target value tk for output unit k to 0.9 if the input class is the kth class,
                    # "0.1 otherwise
                    # t_k = 0.9 if truth == k else 0.1  # had truth == o_k instead of truth == the index

                    # this has to change since this is not binary classification, truths are not 1 and 0
                    # instead each index of the input has a corresponding truth index
                    # need to reduce some of squares error across all output units
                    t_k = truth[k]

                    error = o_k * (1 - o_k) * (t_k - o_k)
                    output_error[k] = error

                # "for each hidden unit j, calculate error term δ_j (k = output units)
                # δ_j <- h_j (1 - h_j) (Σ_k ( w_kj * δ_k ) )
                for j, h_j in enumerate(self.hidden_layer):
                    # calculate sum
                    # total = 0
                    # for k in range(len(self.output_layer)):
                    #    total += self.output_layer_weights[j][k] * output_error[k]
                    total = np.dot(self.output_layer_weights[j], output_error)

                    error = h_j * (1 - h_j) * total
                    # error = self.sigmoid(h_j) * total
                    hidden_error[j] = error  # oops was appending still

                # decrease eta on a schedule: constant for 100 epochs and then small
                if epoch <= 100:
                    schedule_eta = self.eta
                elif epoch <= 200:
                    schedule_eta = self.eta / 5
                else:
                    schedule_eta = self.eta / 10

                # "Hidden to Output layer: For each weight w_kj
                # w_kj = w_kj + Δw_kj
                # Δw_kj = η * δ_k * h_j
                self.output_layer_weights_change = \
                    schedule_eta * (self.hidden_layer.reshape(N+1, 1) @ output_error.reshape(1, self.output_len)) + \
                    self.momentum * self.output_layer_weights_change
                self.output_layer_weights += self.output_layer_weights_change

                # "Input to Hidden layer: For each weight w_ji
                # w_ji = w_ji + Δw_ji
                # Δw_ji = η * δ_j * x_i
                self.hidden_layer_weights_change = \
                    self.eta * (d.reshape(self.input_len, 1) @ hidden_error.reshape(1, N+1)) + \
                    self.momentum * self.hidden_layer_weights_change
                self.hidden_layer_weights += self.hidden_layer_weights_change

            error = np.sum(abs(self.output_layer - truth))

            # add the error to the total accuracy
            # accuracy += np.sum(abs(output_error))
            avg += error

        # average error per data point (individual sample)
        total_size = len(data[0])
        return avg / (total_size * SAMPLES)

    def run(self, data, epochs):
        train_accuracy = []
        test_accuracy = []

        print("Epoch 0: ", end="")
        train_accuracy.append(self.compute_accuracy(data.train(), 0, True))
        test_accuracy.append(self.compute_accuracy(data.test(), 0, True))
        print("Training Set:\tError:", "{:0.5f}".format(train_accuracy[0]), end="\t")
        print("Testing Set:\tError:", "{:0.5f}".format(test_accuracy[0]))

        for i in range(epochs):
            print("Epoch " + str(i + 1) + ": ", end="")
            train_accuracy.append(self.compute_accuracy(data.train(), i))
            test_accuracy.append(self.compute_accuracy(data.test(), i, True))
            print("Training Set:\tError:", "{:0.5f}".format(train_accuracy[i + 1]), end="\t")
            print("Testing Set:\tError:", "{:0.5f}".format(test_accuracy[i + 1]))

            self.save_model(i)  # save the model so we don't always have to do this again

        return train_accuracy, test_accuracy

    def save_model(self, epoch):
        s = str(SAMPLES)
        n = str(N)
        e = str(epoch + 1)
        # saving the model really just means saving the weights
        np.savez("models/samples" + s + "_hidden" + n + "_epoch" + e,
                 hidden=self.hidden_layer_weights, output=self.output_layer_weights)

    def dump_wavs(self, data, prefix):
        sse_avg = 0

        # for each item in the dataset
        for i, (d, truth) in enumerate(zip(data[0], data[1])):
            if i > 4:
                break

            #####################
            # FORWARD PROPAGATION
            #####################

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            self.hidden_layer = np.dot(d, self.hidden_layer_weights)
            self.hidden_layer = np.array([self.sigmoid(x) for x in self.hidden_layer])

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            self.output_layer = np.array([self.sigmoid(x) for x in self.output_layer])

            samples = (d * 2 ** 16 - 2 ** 15).astype(np.int16)
            wf.write("audio/gen/" + prefix + str(i) + "_a.wav", 48000, samples)

            samples = (self.output_layer * 2 ** 16 - 2 ** 15).astype(np.int16)
            wf.write("audio/gen/" + prefix + str(i) + "_b.wav", 48000, samples)

            samples = (truth * 2 ** 16 - 2 ** 15).astype(np.int16)
            wf.write("audio/gen/" + prefix + str(i) + "_c.wav", 48000, samples)


def main():

    d = Data()
    p = NeuralNetwork(ETA, MOMENTUM)

    results = p.run(d, MAX_EPOCHS)

    p.dump_wavs(d.train(), "train")
    p.dump_wavs(d.test(), "test")

    # plot the training / testing accuracy
    plt.plot(list(range(MAX_EPOCHS + 1)), results[0])
    plt.plot(list(range(MAX_EPOCHS + 1)), results[1])
    plt.xlim([0, MAX_EPOCHS])
    # plt.ylim([0, 100])
    plt.show()


if __name__ == '__main__':
    main()
