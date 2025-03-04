# Transfer Learning for Computer Vision with PyTorch

This repository contains a project that implements transfer learning for image classification using PyTorch. It leverages a pretrained convolutional neural network (ResNet18) to classify images into two categories: ants and bees. The project demonstrates two approaches:
- **Fine-Tuning:** The entire network is updated by replacing the final fully connected layer.
- **Fixed Feature Extractor:** All layers except the final classification layer are frozen.

---

## Project Overview

- **Data Loading and Augmentation:**  
  Uses `torchvision.datasets.ImageFolder` for loading images and applies data augmentation on the training set (random resized cropping, horizontal flipping) along with resizing and center cropping for the validation set.

- **Model Setup:**  
  - **Fine-Tuning:**  
    Loads a pretrained ResNet18, replaces the final layer to match the number of output classes, and trains all the parameters.
  - **Fixed Feature Extractor:**  
    Freezes all the layers of the pretrained model except for the final fully connected layer, which is retrained on the new dataset.

- **Training and Evaluation:**  
  Implements a robust training loop that:
  - Processes both training and validation phases.
  - Computes loss and accuracy.
  - Adjusts the learning rate using a scheduler.
  - Saves the best performing model checkpoint.

- **Visualization and Inference:**  
  Provides utility functions to:
  - Display augmented training images.
  - Visualize model predictions on validation images.
  - Run inference on custom images.

---

## Installation

Ensure you have Python 3.6 or later and the following libraries installed:
- PyTorch (with CUDA support if using GPU)
- torchvision
- NumPy
- Matplotlib
- Pillow

Install the dependencies via pip:

```bash
pip install torch torchvision numpy matplotlib pillow
