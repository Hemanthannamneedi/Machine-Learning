# Day 58 – Recurrent Neural Networks (RNN Fundamentals)

## Overview

This project implements a Simple Recurrent Neural Network using TensorFlow and Keras to predict the next number in a numerical sequence.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Model Architecture

- Input Layer
- SimpleRNN
- Dense Output Layer

## Experiments

Three models are trained and compared:

- RNN with 8 hidden units
- RNN with 16 hidden units
- RNN with 32 hidden units

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python day58.py
```

## Evaluation

Models are compared using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Training time

## Outputs

- Trained models in `models/`
- Model comparison: `rnn_comparison.csv`
- Sample prediction: `sample_prediction.csv`
- Evaluation report: `evaluation_report.txt`
- Training loss graph: `plots/training_loss_comparison.png`
- Hidden unit comparison: `plots/hidden_unit_comparison.png`
- Prediction visualization: `plots/prediction_visualization.png`

## Learning Outcomes

- Understanding RNN architecture
- Understanding recurrent cells and hidden states
- Processing sequential data
- Predicting future sequence values
- Comparing different hidden unit sizes
- Understanding vanishing and exploding gradients
