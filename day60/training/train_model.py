
import os
import json
import joblib
import pandas as pd

from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_VERSION = "v1"

FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract"
]

TARGET = "churn"


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df[required].copy()

    df[TARGET] = (
        df[TARGET].astype(str).str.strip().str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    }

    df[TARGET] = df[TARGET].map(mapping)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both churn classes.")

    return df


def build_pipeline():
    numeric_features = [
        "tenure",
        "monthly_charges",
        "support_calls"
    ]

    categorical_features = ["contract"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])


def main():
    print("Loading customer churn dataset...")

    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    if y.value_counts().min() < 2:
        raise ValueError(
            "At least two records per class are needed for "
            "a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = build_pipeline()

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Save complete trained pipeline
    model_path = os.path.join(
        MODEL_DIR,
        f"churn_model_{MODEL_VERSION}.joblib"
    )

    joblib.dump(model, model_path)

    # Verify the saved model
    loaded_model = joblib.load(model_path)

    original_predictions = model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)

    if not (original_predictions == loaded_predictions).all():
        raise RuntimeError("Saved model verification failed.")

    print("Model save/load verification: PASSED")

    metadata = {
        "project": "AI-Powered Customer Churn Prediction",
        "model_name": "Random Forest Classifier",
        "version": MODEL_VERSION,
        "accuracy": float(accuracy),
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_file": f"churn_model_{MODEL_VERSION}.joblib"
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "model_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed successfully!")
    print("Model saved:", model_path)
    print("Metadata saved:", metadata_path)


if __name__ == "__main__":
    main()
