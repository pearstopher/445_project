# CS445 Term Project

Christopher Juncker

Spring 2022

## Instructions

### Generating Audio Data

Before running the program, you will need to generate the datasets
which will be used to train the network.

#### Install
```shell
> python -m pip install pyaudio scipy numpy
```

#### Run
```shell
> python audio/generate.py
```

This command will generate the test and training sets of audio. These
datasets are required for the program to run correctly. PyAudio can be
notoriously difficult to install, you can also download the set of pre-
generated files at 
[pearstopher.com/datasets/445.zip](http://pearstopher.com/datasets/445.zip)


### Training The Network

After you have the datasets, all you need to do is run the program and
let it train the network.

#### Install
```shell
> python -m pip install scipy numpy pandas matplotlib
```

#### Run
```shell
# train on the first dataset (sine & square waves)
> python main.py set1

# train on the second dataset (tbd)
> python main.py set2
```

Running these commands will cause the program to load the specified 
data set and begin training the network. The program will run for a
set number of epochs, and will output the testing and training accuracy
after each epoch.


