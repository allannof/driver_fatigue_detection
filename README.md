# Driver Fatigue Detection

This is a fork of David Buffonis work on the DAIS 7.3 demonstrator. DAIS project participants can collaborate in this fork.

## Project structure
The overall project structure is detailed as follows:
- *dataset*: This directory contains a preprocessed version of the [UTA-RLDD dataset](https://sites.google.com/view/utarldd/home "A dataset with video footage of different stages of drowsiness").
- *ml*: This directory contains machine learning scripts and preprocessing scripts mainly intended to be used client side (to host a personalized ML model on an edge node, according to the federated machine learning architecture)
- *mlops*: This directory contains scripts for running and hosting a shared machine learning model according to the federated machine learning architecture.

Each of these folders have an accompanyiing README, detailing how to setup and run scripts, explaining data content, script purpose, etc.
