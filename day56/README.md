# Day 56 – Artificial Neural Network (ANN)

## Project Overview

This project introduces Deep Learning by implementing an Artificial Neural Network using TensorFlow and Keras for customer churn classification.

## Technologies

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## ANN Architecture

- Input Layer
- One Hidden Layer
- ReLU Activation
- Sigmoid Output Layer
- Adam Optimizer
- Binary Cross-Entropy Loss

## Experiments

The project compares four ANN architectures using 5, 10, 20 and 50 hidden neurons.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python day56.py
```

## Generated Outputs

- `ann_comparison.csv`
- `models/ann_10_neurons.keras`
- `models/preprocessor.joblib`
- `plots/ann_training_loss.png`
- `plots/neuron_comparison.png`

## Learning Outcomes

- Understand ANN architecture.
- Understand neurons and hidden layers.
- Learn activation functions.
- Train neural networks using TensorFlow/Keras.
- Compare different network architectures.

## Note

The project uses a small educational churn dataset. Results are intended for learning and are not representative of production performance.
