# Audio Data Generator
# CS455 Term Project
# Christopher Juncker

import os
import numpy as np
import scipy.io.wavfile as wf
from scipy import signal


# class which creates samples based on specifications
class Sampler:
    def __init__(self,
                 wave="sine",  # or "square"
                 channels=1,
                 bits=16,
                 amplitude=0.5,
                 duration=1.0,
                 frequency=440,
                 sample_rate=48000,
                 ):
        self.wave = wave
        self.channels = channels
        self.bits = bits
        self.amplitude = amplitude
        self.duration = duration
        self.frequency = frequency
        self.sample_rate = sample_rate

        # set max value for current number of bits
        self.max = 2**(bits - 1) - 1
        self.min = -self.max - 1
        # generate the samples
        self.samples = None
        self.generate_samples()

    def generate_samples(self):
        if self.wave == "sine":
            self.samples = ((self.amplitude * self.max) *
                            np.sin((2*np.pi) *
                                   (np.arange(self.sample_rate*self.duration)) *
                                   (self.frequency/self.sample_rate)))

        if self.wave == "square":
            self.samples = ((self.amplitude * self.max) *
                            signal.square(2 * np.pi * self.frequency *
                                          (np.linspace(0, self.duration, self.sample_rate, endpoint=True))))

    # write the samples to disk
    def write(self, file="default.wav"):
        samples = self.samples.astype(np.int16)
        wf.write(file, self.sample_rate, samples)


def main():
    print("Generating Samples")

    # make the directories
    dirs = ('datasets/sine/',
            'datasets/square/',
            )
    for d in dirs:
        exist = os.path.exists(d)
        if not exist:
            os.makedirs(d)

    # generate the files

    # generate sine waves
    # _cut removes lowest 15 frequencies from piano
    frequencies = np.fromfile("frequencies.txt", sep='\n')
    for i, f in enumerate(frequencies):
        s = Sampler("sine", frequency=f)
        s.write(dirs[0] + str(i + 1) + ".wav")

    # generate square waves
    for i, f in enumerate(frequencies):
        s = Sampler("square", frequency=f)
        s.write(dirs[1] + str(i + 1) + ".wav")


if __name__ == '__main__':
    main()
