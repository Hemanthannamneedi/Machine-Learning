# Day 59 – Model Saving, Loading and FastAPI Integration

## Overview

A Customer Churn Prediction API built using Scikit-learn, Joblib and FastAPI.

The project demonstrates how to train and save an ML model, load it into an API service and provide predictions through HTTP endpoints.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic

## Features

- Random Forest model training
- Model saving and loading
- Prediction consistency verification
- Model versioning
- Single customer prediction
- Batch prediction
- In-memory response caching
- Input validation and error handling

## Project Structure

```text
day59/
├── models/
├── train_model.py
├── main.py
├── churn.csv
├── requirements.txt
├── assignment.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

## Start the API

```bash
python -m uvicorn main:app --reload
```

## API Documentation

Open:

http://127.0.0.1:8000/docs

## Main Endpoints

- GET `/`
- GET `/health`
- GET `/model-info`
- POST `/predict`
- POST `/predict/batch`
- GET `/cache-info`
- POST `/cache/clear`

## Model Versioning

Models are stored as versioned Joblib files. A JSON manifest identifies the currently active model.

## Caching

An LRU cache stores results for repeated customer inputs.

## Learning Outcomes

- Saving and loading trained ML models
- Verifying prediction consistency
- Building REST APIs with FastAPI
- Creating single and batch prediction endpoints
- Implementing model versioning
- Implementing response caching
- Validating API input and handling errors
