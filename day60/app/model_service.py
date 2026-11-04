
import os
import json
import joblib
import pandas as pd

from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)


def load_model():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Model metadata not found. Train the model first."
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model_file = metadata["model_file"]
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    loaded_model = joblib.load(model_path)

    return loaded_model, metadata


model, metadata = load_model()


@lru_cache(maxsize=1000)
def cached_prediction(cache_key):
    customer = json.loads(cache_key)

    input_df = pd.DataFrame([customer])

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": prediction,
        "churn_status": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probabilities[1]), 4)
    }


def predict_customer(customer):
    data = {
        "tenure": float(customer.tenure),
        "monthly_charges": float(customer.monthly_charges),
        "support_calls": float(customer.support_calls),
        "contract": customer.contract.strip()
    }

    cache_key = json.dumps(data, sort_keys=True)

    result = cached_prediction(cache_key)

    return {
        **result,
        "model_version": metadata["version"]
    }


def get_model_info():
    return metadata


def get_cache_info():
    return cached_prediction.cache_info()._asdict()


def clear_cache():
    cached_prediction.cache_clear()
