# CS445 Term Project

Christopher Juncker

Spring 2022

## Overview

The purpose of this project is to begin exploring the idea of using Neural
Networks for Digital Signal Modification. I have created this network with
the goal of training it to convert one instrument or type of sound to another.
A more detailed description of the ideas behind the program and the process of
creating it can be found in the program write-up. This readme contains explicit
instructions and examples of how to train and run the network, which were not
included in the program write-up. These instructions involve the use of three
main files included in the repository:
1. `generate.py`
2. `train.py`
3. `convert.py`


## Installation Instructions

In order to run the program you first need to install the required libraries.

```shell
python -m pip install numpy scipy matplotlib
```


## Dataset Generation (`generate.py`)

Before running the main program, you will need to generate the first dataset
of WAV files containing sine and square waves. The generation script is located
in the `audio` directory - just run this command fromm the main directory.

```shell
python audio/generate.py
```

The second dataset has already been generated using sampled instruments provided
by the music production software [Reason](https://www.reasonstudios.com/). This
Dataset is included in the repository and does not need to be generated.


## Training the Network (`train.py`)

In order to train the network on the datasets, simply invoke the training script
and provide the datasets as arguments. The first argument represents the input
values, and the second argument represents the truth values or the desired
output which the network will be attempting to approximate.

1. Train the network to convert square wave samples to sine wave samples:

```shell
python train.py square sine
```

2. Train the network to convert piano samples to stringed instrument samples.

```shell
python train.py piano strings
```

After each epoch of training, the weights for the trained network are saved in
the `models` directory. After the training is complete, any of these models can 
then be loaded and used to convert audio files from one sound to another.


## Converting Files (`convert.py`)

In order to convert audio files with a trained network model, I have provided a 
third and final script. In order to run this script, you simply need to provide
two arguments: a trained model, and a WAV file to be converted.


```shell
python convert.py models/examples/samples500_hidden400_epoch48.npz audio/examples/square.wav
```

The converted file will be located in the `audio/converted` directory, and it 
will have `converted_` prepended to the original filename.


## Usage Examples

Here are a few examples which can be copy/pasted in order to test out the network
without exerting too much effort.

1. Convert a single square wave to a sine wave with a pre-trained model.

```shell
python convert.py models/examples/samples500_hidden400_epoch48.npz audio/examples/square.wav
ls audio/converted/converted_square.wav  # play, analyze, etc
```

2. Use a pre-trained model to convert a melody of notes from square to sine waves.
```shell
python convert.py models/examples/samples500_hidden400_epoch48.npz audio/examples/10s.wav
ls audio/converted/converted_10s.wav  # play, analyze, etc
```

4. Train a fresh network on Dataset #2 and convert a sample piano WAV.
```shell
python train.py piano strings
python convert.py models/samples500_hidden400_epoch50.npz audio/examples/piano_test.wav
ls audio/converted/converted_piano_test.wav  # play, analyze, etc
# the results from this one shouldn't be too surprising after reading the writeup!
```

# Future Goals (`fourier.py`)

As I mentioned in the write-up, I still have a lot of work to do if I want my network
to perform as well on real instruments as it does on sine/square waves. Currently, I am
working on re-creating my program so that it uses the fourier transform as part
of the preprocessing phase. This will enable my network to be able to work directly
on the frequency data. Honestly there is lot more I could add here, my main reason
for stopping short is just that my write-up is already longer than it's supposed to 
be, just accounting for the work which I have accomplished so far! This is something
that I'm going to continue experimenting with over the summer for sure.

Since `fourier.py` is based on `train.py` it can be called similarly:

```shell
python fourier.py piano strings
```

If you run the above command, you can see the network happily begins to train itself
on the fourier transform data which it generates from the WAV files in the dataset. 
The training is pretty successful, as shown briefly at the end of my write-up. 
Currently, I have not finished implementing an updated version of `convert.py`. 
This is because I am still working on solving the problem of how to correctly
resynthesize the modified FFT data. Although I still have a lot to learn, I am 
encouraged that this updated version of the network seems to be performing really 
well on the FFT data so far. If I've learned anything from studying Phase Vocoders
and implementing FFT-based pitch shifting algorithms, it's that taking FFT
data apart is easy, but putting it back together is always tougher than you expect.


