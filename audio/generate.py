# Audio Data Generator
# CS455 Term Project
# Christopher Juncker

import os
import numpy as np
import scipy.io.wavfile as wf


# class which creates samples based on specifications
class Sampler:
    def __init__(self,
                 channels=1,
                 bits=16,
                 amplitude=1.0,
                 duration=1.0,
                 frequency=440,
                 sample_rate=48000,
                 ):
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
        self.samples = ((self.amplitude * self.max) *
                        np.sin((2*np.pi) *
                               (np.arange(self.sample_rate*self.duration)) *
                               (self.frequency/self.sample_rate)))

    # write the samples to disk
    def write(self, file="default.wav"):
        samples = self.samples.astype(np.int16)
        wf.write(file, self.sample_rate, samples)


def main():
    print("Generating Samples")

    # make the directories
    dirs = ('sine/test',
            'sine/train',
            )
    for d in dirs:
        exist = os.path.exists(d)
        if not exist:
            os.makedirs(d)

    # generate the files

    # generate sine waves
    frequencies = np.fromfile("frequencies.txt", sep='\n')
    for i, f in enumerate(frequencies):
        s = Sampler(frequency=f)
        s.write(dirs[0] + "/" + str(i + 1) + ".wav")


if __name__ == '__main__':
    main()
