# Day 53 – ML Pipelines & Handling Class Imbalance

## Overview

This project demonstrates reusable Scikit-Learn pipelines and techniques for handling imbalanced customer churn data.

## Concepts Covered

- Scikit-Learn Pipeline
- Data preprocessing
- Missing-value imputation
- One-hot encoding
- Class imbalance
- `class_weight="balanced"`
- SMOTE
- Precision, Recall and F1-score

## Model Comparison

Three Random Forest approaches are evaluated:

1. Baseline Random Forest
2. Random Forest with balanced class weights
3. SMOTE + Random Forest

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python day53.py
```

## Output

- `model_comparison.csv`
- `plots/imbalance_comparison.png`

## Note

The project uses a small educational churn dataset. Model metrics are illustrative and may vary with the dataset and train/test split.
