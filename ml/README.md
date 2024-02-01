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
which can send local predictions of fatigue to an ML Flow server, according to the [federated learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html "A modified client-server architecture in machine learning, providing low latency and high security") architecture.

### Usage
#### Enabling GPU acceleration
Many machine learning algorithms rely heavily on matrix multiplications and other operations commonly used in computer graphics. Therefore, most (if not all) machine learning frameworks and toolkits allow for using a dedicated graphics card for accelerating such operations. Furthermore, many embedded platforms designed for applications with an AI component come featured with a GPU or a [neural processing unit](https://en.wikipedia.org/wiki/AI_accelerator "A hardware component for accelerating AI related operations.").

If your workstation of choice has a dedicated graphics card, the efficiency of training and prediction algorithms can be significantly boosted by setting up the necessary device drivers for that. This guide will assume you are working with an NVIDIA graphics card with [CUDA](https://developer.nvidia.com/cuda-toolkit "NVIDIA's framework for enabling parallel computation.") capabilities, but another graphics card such as an AMD GPU with [OpenCL](https://developer.nvidia.com/opencl "A low-level programming language which can be used for parallel computing on graphics cards not compatible with CUDA") capabilities will also do.

To install the latest GPU driver this, you will need to know what graphics card you have got. This can be checked in for example the NVIDIA control panel on your computer. Then go to the manufacturer's web page and navigate to their [download site](https://www.nvidia.com/Download/index.aspx) for device drivers. Enter the dedicated graphics card you have got, and download the driver and run the installer. The NVIDIA Windows GPU driver will be available automatically in WSL as a shared object file. Therefore, **the GPU driver should not be installed in WSL**. 

##### Enabling device drivers for camera/multimedia (WSL)
By default, there is no support in the Microsoft WSL kernel for multimedia devices within WSL. In order to do a live extraction of facial features in WSL, there is some necessary setup required to use a camera device.

###### Install usbipd-win (Win)
[usbipd-win](https://github.com/dorssel/usbipd-win) is a tool for sharing USB devices with other machines. It also has means of sharing via Hyper-V, facilitating access of USB devices to virtual machines and WSL. FInd the [latest release](https://github.com/dorssel/usbipd-win/releases) and download the installer. (msi-file)

###### Install Microsoft WSL2 Linux kernel (WSL)
Next, we need to find the correct version of the Linux kernel for your WSL installation. To do that, we need to find out the necessary information about your WSL installation. `cat /proc/version` will list the version of the Linux kernel in use in your WSL installation. Navigate to the [Microsoft WSL2 Linux kernel Github page](https://github.com/microsoft/WSL2-Linux-Kernel/tags) and find the kernel listed by the previous command. Clone that repository into `/usr/src`. Next a few dependencies will need to be done. All that can be done with `apt`. 

`sudo apt upgrade && sudo apt upgrade -y && sudo apt install -y build-essential flex bison libgtk-3-dev libelf-dev libncurses-dev autoconf libudev-dev libtool zip unzip v4l-utils libssl-dev python3-pip cmake git iputils-ping net-tools dwarves guvcview python-is-python3 bc`

Now we run `sudo make menuconfig`. This makefile will launch a graphical wizard for configuring the kernel you have cloned. Use the menu controls and enable the following settings/flags. *Device Drivers > Multimedia support*, *Device Drivers > Multimedia support > Filter media drivers*, *Device Drivers > Multimedia support > Media Device Types > Cameras and video grabbers*, *Device Drivers > Multimedia support > Media Device Types > Video4Linux options > V4L2 sub-device userspace API*, *Device Drivers > Multimedia support > Media Device Types > Media drivers > Media USB adapters*. Save the changes and exit.

Now we run a couple of more makefiles to set up the kernel `sudo make module_install -j$(nproc) && sudo make install -j$(nproc)`

Now we have an appropriate Linux kernel which we will use instead of the default Linux kernel. Copy it to your Windows installation `sudo cp /usr/src/*path to kernel*/vmlinux /mnt/c/Users/*windows username*`.

###### Configure WSL2 (Windows)
In Windows, we need to create a global config file for WSL. Create a file called `wslconfig` in your user directory. Add the following content:
`[wsl2]
kernel=C:\\Users\\*your_windows_username*\\vmlinux`
Restart WSL by quitting your WSL session, open a Powershell window with admin priviliges, and enter `wsl --shutdown`. Wait for about 10 seconds, and start a WSL session. Your new kernel should now be loaded.

###### Attach your camera to the WSL session (Windows)
In the Powershell window you opened, we are going to attach the camera with usbipd. Type `usbipd list` and find the bus id of your camera. Bind the camera with `usbipd bind -b *camera-busid*`. Finally attach the camera with `usbipd attach -w -b *camera-busid*`. Verify that the camera can be found by typing `lsusb` in WSL. Verify that it is identified as a V4L-device by typing `ls /dev`, the camera should be in a link called something like `video0`.

##### Running the scripts
This section will illustrate a way to utilize the scripts to set up a client node on the edge. This can be used to simulate a car with fatigue detection in the scope of the DAIS project.

###### Train the CNN
First we need to train the machine learning model with some of the UTA-RLDD dataset. Make sure to modify the *cnn1d.py* script by changing the file path to the UTA-RLDD dataset, and change the participant id to a valid participant ID number, e.g. 21. Run the *cnn1d.py* script, and it should output some checkpoints of the training. Let it run for at least 50 epochs to get a decent loss, but running the script all the way shouldn't be a problem.

###### Run live feature extraction
A great way to see how the model behaves is to run a live feature extraction. This is done by running the *extract_features.py* script. Make sure to pass `0` as an argument to the OpenCV `VideoCapture` object, to use the default camera. The script will loop for a few thousand iterations to load a frame buffer, and then do the video analysis, and pass the extracted landmarks to the CNN model you've trained. (make sure to set a correct file path to one of the checkpoints for the model you trained)
