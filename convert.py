# CS445 Term Project
# Christopher Juncker
#
# This file is used to run a specified WAV file through a trained network
# The reconstructed modified samples are saved in audio/converted/.
#
# Arg 1: name of model (models/my_model.npz)
# Arg 2: name of WAV (audio/my_wav.wav)
#

import os
import numpy as np
import scipy.io.wavfile as wf
import sys


# load the model
model = np.load(sys.argv[1])
hidden_layer_weights = model['hidden']
output_layer_weights = model['output']
# I could deduce the number of samples from the weights,
# but for now I'm just getting it from the filename
SAMPLES = int(''.join(filter(str.isdigit, sys.argv[1].split("_")[0])))


# load the wav
sample_rate, samples = wf.read(sys.argv[2])
# scale the samples to be in the range (0, 1)
samples = (samples + 2**15) / 2**16


# number of hidden units
N = len(hidden_layer_weights) - 1  # (bias is extra 1)


# stripped neural network (forward propagation only)
class NeuralNetwork:
    def __init__(self):
        print("Converting WAV...")

        self.input_len = SAMPLES
        self.output_len = SAMPLES

        self.hidden_layer_weights = hidden_layer_weights
        self.hidden_layer = np.zeros((N+1))
        self.hidden_layer[0] = 1  # bias
        self.output_layer_weights = output_layer_weights
        self.output_layer = np.zeros(self.output_len)

    # The activation function for each output unit is the sigmoid function
    # σ(z) = 1 / ( 1 + e^(-z) )
    @staticmethod
    def sigmoid(array):
        array = 1 / (1 + np.e ** (- array))
        return array

    # The activation function for each hidden unit is the Leaky ReLU function
    @staticmethod
    def leakyrelu(array):
        array[array < 0] *= 0.05
        return array

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
            self.hidden_layer = self.leakyrelu(self.hidden_layer)

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            self.output_layer = self.sigmoid(self.output_layer)

            # add the output to the data
            new_data[start:end] += self.output_layer
            # increment the window bounds
            start += SAMPLES//2
            end += SAMPLES//2

        # double the first and last window halves that didn't have overlap
        new_data[-SAMPLES // 2:SAMPLES//2] *= 2

        # now we need to take the average of all the samples (dividing all by 2)
        # but we also need to double the samples to scale them correctly! (multiply all by 2)

        # all we actually have to do then is to simply re-center the samples around 0
        new_data -= 1
        # and call it a day
        return new_data


# smooth the output samples just a little bit to reduce extreme high-frequency noise
def smooth(result):
    smoothness = 3
    result = np.convolve(result, np.ones(smoothness)/smoothness, mode='valid')
    return result


def main():

    p = NeuralNetwork()

    result = p.run(samples)

    result = smooth(result)

    filename = sys.argv[2].split('/')[-1]

    if not os.path.exists('audio/converted'):
        os.makedirs('audio/converted')

    wf.write("audio/converted/converted_" + filename, 48000, result)


if __name__ == '__main__':
    main()
