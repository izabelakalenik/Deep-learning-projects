# Deep learning projects

This repository contains two projects developed for the Deep Learning course, realized in a group of 3. Projects descriptions can be found in [docs](./docs) directory.

## Overview

### 1. Computer Vision & Fundamentals
Implementation of classic and convolutional neural networks for image recognition.

* **Tasks**: Handwritten letter classification (EMNIST) and medical cell classification (BloodMNIST).

* **Architectures**: Perceptron, Logistic Regression (manual SGD), Multi-Layer Perceptron (backprop from scratch), and CNNs (PyTorch).

* **Key focus**: Impact of hyperparameters, activation functions (Softmax vs. Sparsemax), and pooling layers on performance.

### 2. Genomic Sequence Modeling (RBPs Interaction)
Predicting RNA-binding protein (RBP) interaction using advanced sequence modeling.

* **Task**: Regression problem to predict binding affinity (affinity) for the RBFOX1 protein.

* **Architectures**: Implementation and comparison of CNN, RNN (LSTM/GRU), and Transformers.

* **Advanced Features**: Integration of Self-Attention mechanisms and Attention-pooling to improve spatial dependency learning.

* **Metrics**: Optimized using masked MSE loss and evaluated via Spearman Rank Correlation.

## Technologies
* Python

* PyTorch, NumPy, MedMNIST

* Matplotlib (visualization), Scipy (Spearman correlation)
