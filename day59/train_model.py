
import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone

# Configuration
DATA_PATH = "churn.csv"
MODEL_DIR = "models"
MODEL_VERSION = "v1"

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract"
]

TARGET = "churn"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

df = df[required_columns].copy()

# 2. Clean target
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

target_mapping = {
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1
}

df[TARGET] = df[TARGET].map(target_mapping)
df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

if df[TARGET].nunique() < 2:
    raise ValueError("The dataset must contain both churn classes.")

X = df[FEATURES]
y = df[TARGET]

# 3. Define preprocessing
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
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 4. Build complete ML pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 5. Evaluate using a holdout test set
class_counts = y.value_counts()

if len(class_counts) == 2 and class_counts.min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
else:
    raise ValueError(
        "At least two examples per churn class are required "
        "for a stratified train/test split."
    )

evaluation_pipeline = clone(pipeline)
evaluation_pipeline.fit(X_train, y_train)

test_predictions = evaluation_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# 6. Train final model on all available data
pipeline.fit(X, y)

# 7. Check predictions before saving
sample = X.iloc[:5].copy()
predictions_before_save = pipeline.predict(sample)

# 8. Save versioned model
model_path = os.path.join(
    MODEL_DIR,
    f"churn_model_{MODEL_VERSION}.joblib"
)

joblib.dump(pipeline, model_path)

# 9. Load saved model and verify predictions
loaded_model = joblib.load(model_path)
predictions_after_load = loaded_model.predict(sample)

if not np.array_equal(
    predictions_before_save,
    predictions_after_load
):
    raise RuntimeError(
        "Predictions changed after saving and loading the model."
    )

print("\nSave/load verification: PASSED")
print("Predictions before saving:", predictions_before_save)
print("Predictions after loading:", predictions_after_load)

# 10. Save model version metadata
metadata = {
    "model_name": "Customer Churn Prediction",
    "version": MODEL_VERSION,
    "model_file": f"churn_model_{MODEL_VERSION}.joblib",
    "algorithm": "Random Forest Classifier",
    "features": FEATURES,
    "test_accuracy": float(accuracy),
    "trained_at_utc": datetime.now(timezone.utc).isoformat()
}

with open(
    os.path.join(MODEL_DIR, "model_metadata.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

# 11. Save current model manifest
manifest = {
    "current_version": MODEL_VERSION,
    "model_path": model_path
}

with open(
    os.path.join(MODEL_DIR, "current_model.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(manifest, file, indent=4)

print("\nModel training completed successfully!")
print("Saved model:", model_path)
print("Saved metadata: models/model_metadata.json")
print("Current version:", MODEL_VERSION)
