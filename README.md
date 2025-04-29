# Project_MIRCV
**DESCRIPTION**  
  
Graphical interface in python for training CNN models and visual proof of the classification of test set images. 
Interface with the task of putting most of the parameters required for training into visual mode and displaying the results. 
The program will also generate command line prints and plots for additional information.

**The folder contains:**  

The folders:
- dataset: 
- model:

The python files:
- genCNNClassifier:
- 
 

**HW used**  

CPU: Intel(R) Core(TM) i7-10870H CPU @ 2.20GHz   2.21 GHz
RAM: 16 GB
GPU: RTX 3060 6GB laptop

**Settings**

Python and tensorflow are used to run this codce and train the networks. 
Caution: tensorflow's setting up with the nvidia drivers to use the GPU and speed up the training process is very delicate.
I refer to the official guide at the following link: https://www.tensorflow.org/install/pip?hl=it#windows-native
Attention on windows last tensorflow release with GPu support is 2.10.
For this project I have doperated two configurations, an initial one on windows and then a more advanced one using wsl.

Windows configuration:
python = 3.10, tensorflow = 2.10, CUDA = 11.2, cuDNN = 8.1

Wsl configuration (via miniconda)
python = 3.10, tensorflow = 2.19, CUDA = 12.9, cuDNN = cuDNN 9.3

I have listed my configurations for clarity, do not consider them the best or those to be replicated, check according to your environment and your needs the best setting for you.

**Execution Guide**  
	    

**Developer's notes**  

This project has no academic purpose but is just for fun and personal growth. For these reasons it will not have too detailed documentation or presentation.

**Developers:**  
- Alessandro Diana