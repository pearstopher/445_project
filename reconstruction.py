# CS445 Term Project
# Christopher Juncker
#
# This file is used to run a specified WAV file through a trained network
# The reconstructed modified samples are saved in audio/reconstructed/.
#
# Arg 1: name of model (models/my_model.npz)
# Arg 2: name of WAV (audio/my_wav.wav)
#

import numpy as np
import scipy.io.wavfile as wf
import sys
import scipy.signal


# load the model
model = np.load(sys.argv[1])
hidden_layer_weights = model['hidden']
output_layer_weights = model['output']
SAMPLES = 500


# load the wav
sample_rate, samples = wf.read(sys.argv[2])
# scale the samples to be in the range (0, 1)
samples = (samples + 2**15) / 2**16


# number of hidden units
N = len(hidden_layer_weights) - 1  # (bias is extra 1)


# stripped neural network (forward propagation only)
class NeuralNetwork:
    def __init__(self):
        print("Initializing neural network...")

        self.input_len = SAMPLES
        self.output_len = SAMPLES

        self.hidden_layer_weights = hidden_layer_weights
        self.hidden_layer = np.zeros((N+1))
        self.hidden_layer[0] = 1  # bias
        self.output_layer_weights = output_layer_weights
        self.output_layer = np.zeros(self.output_len)

    # "The activation function for each hidden and output unit is the sigmoid function
    # σ(z) = 1 / ( 1 + e^(-z) )
    @staticmethod
    def sigmoid(array):
        array = 1 / (1 + np.e ** (- array))
        return array

    # "Compute the accuracy on the training and test sets for this initial set of weights,
    # "to include in your plot. (Call this “epoch 0”.)
    #
    # data[0] = input data
    # data[1] = truth data
    def run(self, data):
        new_data = np.zeros_like(data)

        # loop through the data in overlapping SAMPLES sized chunks
        start = 0
        end = start + SAMPLES
        while end <= len(data):
            d = data[start:end]

            #####################
            # FORWARD PROPAGATION
            #####################

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            self.hidden_layer = np.dot(d, self.hidden_layer_weights)
            self.hidden_layer = self.sigmoid(self.hidden_layer)

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            self.output_layer = self.sigmoid(self.output_layer)

            new_data[start:end] += self.output_layer

            start += SAMPLES//2
            end += SAMPLES//2

        # take the average of all of the doubled-up samples
        new_data[SAMPLES//2:-SAMPLES//2] /= 2
        # center the samples
        new_data -= 0.5
        new_data *= 2

        return new_data


# really basic fft-based noise reduction
# not called since it doesn't really work
def smooth(result):
    fs = 48000
    f, t, zxx = scipy.signal.stft(result, fs=fs, nperseg=fs//2)

    # reduce the amplitudes
    zxx -= 0.001
    # zero out negatives
    zxx[zxx < 0] = 0
    # raise the amplitudes back up
    zxx[zxx > 0] += 0.001

    _, result = scipy.signal.istft(zxx, fs)
    return result


def main():

    p = NeuralNetwork()

    result = p.run(samples)

    # result = smooth(result)

    filename = sys.argv[2].split('/')[-1]

    wf.write("audio/reconstructed/reconstructed_" + filename, 48000, result)


if __name__ == '__main__':
    main()
