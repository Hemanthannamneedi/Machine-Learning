# AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning application that predicts customer churn using a Random Forest Classifier and provides predictions through a FastAPI REST API.

## Project Overview

This application demonstrates the complete ML workflow:

1. Dataset preparation and preprocessing
2. Machine Learning model training
3. Model evaluation and persistence
4. FastAPI backend integration
5. Single and batch predictions
6. Input validation and error handling
7. Prediction caching

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic

## Project Structure

```text
ai-powered-app/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
│
├── models/
│   ├── churn_model_v1.joblib
│   └── model_metadata.json
│
├── training/
│   └── train_model.py
│
├── data/
│   └── churn.csv
│
├── requirements.txt
├── README.md
├── assignment.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-powered-app.git
cd ai-powered-app
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

```bash
python training/train_model.py
```

The trained model and metadata are saved in the `models/` directory.

## Start the API

```bash
python -m uvicorn app.main:app --reload
```

The API runs at:

`http://127.0.0.1:8000`

## API Documentation

Interactive Swagger documentation:

`http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint         | Description             |
| ------ | ---------------- | ----------------------- |
| GET    | `/`              | Application information |
| GET    | `/health`        | Health status           |
| GET    | `/model-info`    | Model metadata          |
| POST   | `/predict`       | Single prediction       |
| POST   | `/predict/batch` | Batch prediction        |
| GET    | `/cache-info`    | Cache statistics        |
| POST   | `/cache/clear`   | Clear cache             |

## Sample Prediction Request

```json
{
  "tenure": 12,
  "monthly_charges": 75.5,
  "support_calls": 3,
  "contract": "Monthly"
}
```

## Example Batch Request

```json
{
  "customers": [
    {
      "tenure": 12,
      "monthly_charges": 75.5,
      "support_calls": 3,
      "contract": "Monthly"
    },
    {
      "tenure": 36,
      "monthly_charges": 45,
      "support_calls": 1,
      "contract": "Yearly"
    }
  ]
}
```

## Model Details

- Algorithm: Random Forest Classifier
- Target: Customer churn
- Output: Churn prediction and estimated churn probability
- Model storage: Joblib
- Model version: v1

## Error Handling

Pydantic validates incoming data. Invalid values generate HTTP 422 responses. Unexpected prediction errors are handled by the API.

## Caching

An in-memory LRU cache stores repeated prediction results to avoid unnecessary inference for identical customer inputs.

## Learning Outcomes

- Building an end-to-end ML application
- Organizing code using clean architecture
- Saving and loading trained models
- Integrating ML models with FastAPI
- Creating REST API endpoints
- Handling invalid requests
- Implementing batch prediction and caching

## Future Improvements

- Add a web frontend
- Deploy the API to a cloud platform
- Add authentication
- Use persistent caching
- Monitor model performance and data drift

## Disclaimer

This project is an educational demonstration. The dataset is small and model predictions should not be treated as production-grade customer churn assessments.
