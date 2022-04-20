# CS410 Computers, Sound & Music
# Homework #1
# Christopher Juncker

import numpy as np

# wave was suggested but I am exploring alternatives like scipy
# import wave
#
# example of writing a .wav file with scipy
# https://stackoverflow.com/questions/52477889
import scipy.io.wavfile as wf


# INSTALLING/IMPORTING sounddevice:
#   (sounddevice was suggested as a simpler alternative to pyaudio)
#   >pip install sounddevice
#   import sounddevice
#
# INSTALLING/IMPORTING pyaudio:
#   suggestion to install visual c++ before installing pyaudio
#   https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
#
#   the above didn't work so I tried installing a 'wheel' for my version of python here:
#   https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
#   then I `pip install The_file_I_downloaded.whl`
#   (source: https://stackoverflow.com/questions/59467023)
import pyaudio


# The Assignment
#
#   1.  Build a program in your chosen language that writes a sine wave to a WAV file named sine.wav
#       in the current directory. Specifications for your sine wave:
#
#           Channels per frame: 1 (mono)
#           Sample size: 16 bits
#           Amplitude: ¼ maximum possible amplitude (-8192..8192)
#           Duration: one second
#           Frequency: 440Hz
#           Sample Rate: 48000 samples per second
#
#       Generate the samples yourself using your language's sin() function — if your language has no
#       math library, get a different language.
#
#   2.  Extend your program to also write a WAV file named clipped.wav in the current directory.
#       Generate a sine wave as though it were ½ maximum amplitude (-16384..16384), except: samples
#       that would be greater than ¼ maximum amplitude (8192) should instead be ¼ maximum amplitude;
#       samples that would be less than ¼ minimum amplitude (-8192) should instead be ¼ minimum
#       amplitude. Other than this change, all other parameters remain the same as in Part 1.
#
#       Your wave should look "clipped" like this in Audacity. Note the phase: the waveform starts
#       upward from zero.
#
#       Also check the sample rate, amplitude and frequency while you're in Audacity. On Linux the
#       file command will also tell you the frequency, number of channels, and bits per sample. Or
#       you can use the wavfile Python utility I built for the course. Lots of options.
#
#   3.  Extend your program to also play the same clipped sine wave on your computer's audio output.
#       Do not shell out to an external program for this: use an audio library.
#
#       Each time your program is run, it should do all three of these things.
#


# class for holding the specifications for the audio file
class Spec:
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


# class which creates samples based on Spec
class Sampler(Spec):
    def __init__(self, channels, bits, amplitude, duration, frequency, sample_rate):
        super().__init__(channels, bits, amplitude, duration, frequency, sample_rate)
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

    # play the samples with pyaudio
    def play(self):
        # this should only be used when bytes are set to 16
        # don't convert the original samples, they will be reused
        samples = self.samples.astype(np.int16)  # np.Float32, etc

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,  # paFloat32, etc
                        channels=1,
                        rate=self.sample_rate,
                        output=True)
        # https://stackoverflow.com/questions/8299303 yahweh - tobytes()
        stream.write(1 * samples.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()

    # write the samples to disk
    def write(self, file="default.wav"):
        samples = self.samples.astype(np.int16)
        wf.write(file, self.sample_rate, samples)


def main():
    print("Homework 1")

    # Part 1
    #
    # Channels per frame: 1 (mono)
    # Sample size: 16 bits
    # Amplitude: ¼ maximum possible amplitude (-8192..8192)
    # Duration: one second
    # Frequency: 440Hz
    # Sample Rate: 48000 samples per second
    s = Sampler(1, 16, 0.25, 1, 440, 48000)
    s.play()
    s.write("sine.wav")

    # Part 2
    #
    # ½ maximum amplitude (-16384..16384), except:
    # samples that would be greater than ¼ maximum amplitude (8192) should instead be ¼ maximum amplitude;
    # samples that would be less than ¼ minimum amplitude (-8192) should instead be ¼ minimum amplitude.

    # update amplitude
    # could make a new Sampler but why not re-use the old one?
    s.amplitude = 0.5
    s.generate_samples()

    # chop the top
    new_max = s.max / 4
    s.samples = np.where(s.samples <= new_max, s.samples, new_max)

    # chop the bottom
    new_min = s.min / 4
    s.samples = np.where(s.samples >= new_min, s.samples, new_min)

    s.play()
    s.write("clipped.wav")


if __name__ == '__main__':
    main()
