# CCSNe_Detection
CCSNe detection project code from paper https://arxiv.org/abs/2410.06430

"train_network_m.m" is the main Matlab script for training the CNN using the Q-transform output spectrogram. It runs "initialize_network.m" to initialize the ResNet18 network.

The pre-trained network parameters "initial_params_resnet18.mat" can be found here https://drive.google.com/file/d/1ecVfsLRXvlYvILDh2OV4FR0ZAj5b8B2E/view?usp=drive_link.

The Python script "Generate_qgrams_signal_latest.ipynb" generates and saves the Q-transform spectrograms as images. 
