# Day 57 – Convolutional Neural Network (CNN)

## Overview

This project implements a Convolutional Neural Network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## CNN Architecture

- Input Layer
- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Dropout
- Softmax Output Layer

## Dataset

The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python day57.py
```

## Outputs

- Trained CNN model: `models/mnist_cnn.keras`
- Evaluation report: `evaluation_report.txt`
- Accuracy graph: `plots/accuracy_curve.png`
- Loss graph: `plots/loss_curve.png`
- Sample predictions: `plots/sample_predictions.png`
- Confusion matrix: `plots/confusion_matrix.png`

## Evaluation

The model is evaluated using test accuracy, test loss, classification report, and visualized predictions.

## Learning Outcomes

- Understanding convolution operations
- Understanding max pooling
- Preparing image datasets
- Building CNN architectures
- Training and evaluating image classification models
- Visualizing model predictions
