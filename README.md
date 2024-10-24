# CCSNe_Detection
CCSNe detection project code from paper https://arxiv.org/abs/2410.06430

To run the demo for prediction:

1. Run the CCSNE_demo.ipynb with the necessary packages installed
2. You need to download the "CCSNe_QNet_24Jan.onnx" from here https://drive.google.com/file/d/1gjYxJsi-Q4HSfraSWTjp_UxQdJzTMx3R/view?usp=drive_link

To train the model yourself:

1. The Python script "Generate_qgrams_signal_latest.ipynb" generates and saves the Q-transform spectrograms as images. "s11.2--LS220_0.1kpc_sim100_SNR29.21.txt" is an example signal time series as the input, and "s11.2--LS220_0.1kpc_sim100_SNR29.21.png" is the Q-transform output saved as an image. The same process applies for the LIGO noise, but instead of "new_simulation_training/signal/" one should use "new_simulation_training/noise/".

2. "train_network_m.m" is the main Matlab script for training the CNN using the Q-transform output spectrogram. It runs "initialize_network.m" to initialize the ResNet18 network.

3. The pre-trained network parameters "initial_params_resnet18.mat" can be downloaded here https://drive.google.com/file/d/1ecVfsLRXvlYvILDh2OV4FR0ZAj5b8B2E/view?usp=drive_link.

4. Alternatively you could also use "CCSNe_detection_training.ipynb" for network training.

