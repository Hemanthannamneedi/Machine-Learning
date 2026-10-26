# House Price Prediction — End-to-End ML Pipeline

## Overview

This project predicts house prices using a complete Machine Learning pipeline built with Python and Scikit-learn.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Workflow

1. Load and inspect the housing dataset.
2. Remove duplicate records and invalid target values.
3. Handle missing numerical and categorical features.
4. Encode categorical data using One-Hot Encoding.
5. Split the dataset into training and testing sets.
6. Train a Linear Regression model.
7. Evaluate performance using MAE, MSE, RMSE, and R².
8. Apply five-fold cross-validation.
9. Visualize actual versus predicted prices.
10. Save and reload the trained pipeline using Joblib.
11. Predict the price of a new house.

## Run the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python main.py
```

## Output Files

- `models/house_model.joblib` — saved preprocessing and prediction pipeline.
- `plots/actual_vs_predicted.png` — prediction comparison chart.

## Dataset

The included dataset is a small illustrative dataset for educational purposes. Its results are not representative of real estate market performance.

## Learning Outcome

This project demonstrates an end-to-end regression workflow, model evaluation, cross-validation, model persistence, and prediction using a saved model.
