## About
This directory contains scripts with machine learning models for the Driver Fatigue Detection system, as well as other scripts
intended to be run client side, i.e. inside a driver's car.

### Key scripts
- *cnn1d.py*: A script containing a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network "neural network architecture using kernel operations/convolutions to reduce number of neurons needed")
for driver fatigue detection, as well as functions for training that model on some data, and storing checkpoints from training.
- *extract_features.py*: A script using [OpenCV2](https://docs.opencv.org/4.9.0/d1/dfb/intro.html "tool for capturing images/video, with support for image analysis") to
record facial landmarks with the purpose of extracting features related to fatigue. The script can be run with a prerecorded video as input, or it can perform live extraction.
The script will output a binary classification of alert/sleepy.
- *data_preprocessing.py*: A script which accepts CSV-files with values of facial features, the target feature and a participant ID, and bundles them in Pickle files.
The idea is to run this script with an input path and participant ID as parameters, to bundle training and test data for that participant, within two Pickle files instead of six CSV files.
- *register.py*: This script starts an [ML Flow](https://mlflow.org/ "A framework for hosting a machine learning model in a client-server architecture, providing common ML Ops operations") client,
which can send local predictions of fatigue to an ML Flow server, according to the [federated learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html "A modified client-server architecture in machine learning, providing low latency and high security") architecture

### Usage
1. Enabling GPU acceleration
2. Enabling device drivers for camera/multimedia (WSL)
3. Running the scripts
