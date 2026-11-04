
import os
import json
import hashlib
import joblib
import pandas as pd

from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = "models"
MANIFEST_PATH = os.path.join(
    MODEL_DIR,
    "current_model.json"
)

FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract"
]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML model service built with FastAPI",
    version="1.0.0"
)

# 1. Load the currently selected model
try:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    MODEL_VERSION = manifest["current_version"]
    MODEL_PATH = manifest["model_path"]

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

except Exception as error:
    raise RuntimeError(
        f"Unable to load the ML model: {error}"
    ) from error


# 2. Define input validation schema
class CustomerInput(BaseModel):
    tenure: float = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_calls: float = Field(ge=0)
    contract: str = Field(min_length=1, max_length=50)


class BatchPredictionInput(BaseModel):
    customers: list[CustomerInput] = Field(
        min_length=1,
        max_length=100
    )


# 3. Prediction cache
@lru_cache(maxsize=1000)
def cached_prediction(cache_key):
    customer_data = json.loads(cache_key)

    input_df = pd.DataFrame(
        [customer_data],
        columns=FEATURES
    )

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]
    churn_probability = float(probabilities[1])

    return {
        "prediction": prediction,
        "churn_status": (
            "Yes" if prediction == 1 else "No"
        ),
        "churn_probability": round(
            churn_probability, 4
        )
    }


def make_cache_key(customer):
    normalized_data = {
        "tenure": float(customer.tenure),
        "monthly_charges": float(customer.monthly_charges),
        "support_calls": float(customer.support_calls),
        "contract": customer.contract.strip()
    }

    serialized = json.dumps(
        normalized_data,
        sort_keys=True
    )

    return serialized


def predict_customer(customer):
    try:
        cache_key = make_cache_key(customer)
        result = cached_prediction(cache_key)

        return {
            **result,
            "model_version": MODEL_VERSION
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# 4. Root endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "model_version": MODEL_VERSION,
        "docs": "/docs"
    }


# 5. Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }


# 6. Model information endpoint
@app.get("/model-info")
def model_info():
    metadata_path = os.path.join(
        MODEL_DIR,
        "model_metadata.json"
    )

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        return metadata

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read model metadata: {str(error)}"
        )


# 7. Single prediction endpoint
@app.post("/predict")
def predict(customer: CustomerInput):
    return predict_customer(customer)


# 8. Batch prediction endpoint
@app.post("/predict/batch")
def batch_predict(request: BatchPredictionInput):
    try:
        results = [
            predict_customer(customer)
            for customer in request.customers
        ]

        return {
            "total_predictions": len(results),
            "model_version": MODEL_VERSION,
            "predictions": results
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(error)}"
        )


# 9. Cache information endpoint
@app.get("/cache-info")
def cache_info():
    return cached_prediction.cache_info()._asdict()


# 10. Clear prediction cache
@app.post("/cache/clear")
def clear_cache():
    cached_prediction.cache_clear()

    return {
        "message": "Prediction cache cleared successfully"
    }
    