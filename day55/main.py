
import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("churn.csv")

print("Dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset columns:", df.columns.tolist())


# Clean column names
df.columns = df.columns.str.strip().str.lower()

required_columns = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "churn"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: " + str(missing_columns)
    )


# Convert target to numeric
df["churn"] = (
    df["churn"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"no": 0, "yes": 1})
)

df = df.dropna(subset=["churn"]).copy()
df["churn"] = df["churn"].astype(int)


# ==================================================
# 2. FEATURE ENGINEERING
# ==================================================

# Total estimated customer charges
df["total_charges"] = (
    df["monthly_charges"] * df["tenure"]
)

# Support calls relative to customer tenure
df["support_call_rate"] = (
    df["support_calls"] / (df["tenure"] + 1)
)

# Create customer tenure groups
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 6, 12, 24, np.inf],
    labels=[
        "New",
        "Short Term",
        "Medium Term",
        "Long Term"
    ]
)

print("\nFeature engineering completed.")
print(df.head())


# ==================================================
# 3. DEFINE FEATURES AND TARGET
# ==================================================

X = df.drop(columns=["churn"])
y = df["churn"]

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

print("\nChurn distribution:")
print(y.value_counts())


# ==================================================
# 4. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

minority_train_count = int(y_train.value_counts().min())

if minority_train_count < 2:
    raise ValueError(
        "SMOTE requires at least two training samples "
        "in the minority class. Use a larger dataset."
    )

cv_splits = min(3, minority_train_count)

cv = StratifiedKFold(
    n_splits=cv_splits,
    shuffle=True,
    random_state=42
)

smote_neighbors = min(5, minority_train_count - 1)


# ==================================================
# 5. PREPROCESSING PIPELINE
# ==================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ==================================================
# 6. DEFINE BASELINE MODELS
# ==================================================

base_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=3
    ),

    "SVM": SVC(
        kernel="rbf"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )
}


# ==================================================
# 7. TRAIN BASELINE MODELS
# ==================================================

results = []
trained_models = {}

for name, classifier in base_models.items():

    print("\nTraining baseline:", name)

    pipeline = ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=smote_neighbors
                )
            ),
            ("model", classifier)
        ]
    )

    start_time = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    predictions = pipeline.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Training Time": training_time
    }

    results.append(metrics)
    trained_models[name] = pipeline

    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("F1 Score:", round(metrics["F1 Score"], 4))


# ==================================================
# 8. HYPERPARAMETER TUNING
# ==================================================

tuning_grids = {
    "Random Forest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, None]
    },

    "Gradient Boosting": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.05, 0.1]
    },

    "XGBoost": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3]
    }
}

tuned_models = {}
tuning_results = []

for name, parameter_grid in tuning_grids.items():

    print("\nGridSearchCV tuning:", name)

    classifier = base_models[name]

    pipeline = ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=42,
                    k_neighbors=smote_neighbors
                )
            ),
            ("model", classifier)
        ]
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    start_time = time.perf_counter()
    grid.fit(X_train, y_train)
    tuning_time = time.perf_counter() - start_time

    tuned_model = grid.best_estimator_
    tuned_models[name + " (Tuned)"] = tuned_model

    predictions = tuned_model.predict(X_test)

    metrics = {
        "Model": name + " (Tuned)",
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Training Time": tuning_time
    }

    results.append(metrics)

    tuning_results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "Best CV F1": grid.best_score_
    })

    print("Best parameters:", grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))
    print("Test F1:", round(metrics["F1 Score"], 4))


# ==================================================
# 9. MODEL LEADERBOARD
# ==================================================

leaderboard = pd.DataFrame(results)

leaderboard = leaderboard.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\nFINAL MODEL LEADERBOARD")
print(leaderboard.to_string(index=False))

leaderboard.to_csv(
    "model_leaderboard.csv",
    index=False
)

pd.DataFrame(tuning_results).to_csv(
    "tuning_results.csv",
    index=False
)


# ==================================================
# 10. SELECT AND SAVE BEST MODEL
# ==================================================

best_model_name = leaderboard.iloc[0]["Model"]

if best_model_name.endswith(" (Tuned)"):
    best_model = tuned_models[best_model_name]
else:
    best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/best_churn_model.joblib"
)

print("\nBest model:", best_model_name)
print("Saved model: models/best_churn_model.joblib")


# ==================================================
# 11. WINNER EVALUATION
# ==================================================

best_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, best_predictions)
precision = precision_score(
    y_test, best_predictions, zero_division=0
)
recall = recall_score(
    y_test, best_predictions, zero_division=0
)
f1 = f1_score(
    y_test, best_predictions, zero_division=0
)

print("\nWINNING MODEL CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ==================================================
# 12. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"]
)

plt.title("Winning Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig("plots/confusion_matrix.png")
plt.close()


# ==================================================
# 13. FEATURE IMPORTANCE
# ==================================================

fitted_preprocessor = best_model.named_steps["preprocessing"]
fitted_classifier = best_model.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()

if hasattr(fitted_classifier, "feature_importances_"):

    importances = fitted_classifier.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    feature_importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    top_features = feature_importance_df.head(10)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Churn Predictors")
    plt.tight_layout()

    plt.savefig("plots/feature_importance.png")
    plt.close()

else:
    feature_importance_df = pd.DataFrame(
        columns=["Feature", "Importance"]
    )
    print(
        "\nThe selected model does not provide "
        "built-in feature importance."
    )


# ==================================================
# 14. LEADERBOARD CHART
# ==================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=leaderboard,
    x="Model",
    y="F1 Score"
)

plt.title("Customer Churn Model Comparison")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("plots/model_leaderboard.png")
plt.close()


# ==================================================
# 15. AUTOMATIC EVALUATION REPORT
# ==================================================

report = f"""
# Advanced Customer Churn Prediction - Evaluation Report

## 1. Executive Summary

The best model based on test-set F1-score was **{best_model_name}**.

- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-score: {f1:.4f}

These results describe performance on the held-out test set.
They should not be treated as guaranteed business outcomes.

## 2. Data Journey

The churn dataset was cleaned and prepared.

Feature engineering included:
- Total charges = Monthly charges multiplied by tenure.
- Support call rate = Support calls divided by tenure plus one.
- Tenure groups: New, Short Term, Medium Term and Long Term.

Numeric features were imputed and scaled.
Categorical features were imputed and one-hot encoded.

SMOTE was applied inside the training pipeline.
The test data was not oversampled.

## 3. Model Leaderboard

{leaderboard.to_string(index=False)}

## 4. Winning Model - Confusion Matrix

- True Negatives: {tn}
- False Positives: {fp}
- False Negatives: {fn}
- True Positives: {tp}

The model correctly identified {tp} churn customers.
It incorrectly flagged {fp} customers who stayed as churn risks.
It missed {fn} actual churn customers.

## 5. Feature Importance

"""

if not feature_importance_df.empty:
    report += feature_importance_df.head(10).to_string(index=False)
else:
    report += (
        "The selected model does not expose built-in feature "
        "importance. Use permutation importance for further analysis."
    )

report += """

## 6. Limitations

- This project uses a small educational dataset.
- Test metrics may change substantially with different data splits.
- SMOTE after one-hot encoding can create fractional encoded
  categorical values; SMOTENC is a better choice for mixed
  numerical and categorical data in production.
- Feature importance indicates model reliance, not causation.
- A larger, representative dataset and further validation are
  needed before using this model for business decisions.

## 7. Conclusion

The model was selected using test-set F1-score.
The saved pipeline includes preprocessing, SMOTE and the
trained classifier, allowing consistent predictions on new data.
"""

with open(
    "evaluation_report.md",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ==================================================
# 16. FINAL SUMMARY
# ==================================================

print("\nPROJECT COMPLETED SUCCESSFULLY")

print("Best model:", best_model_name)
print("F1 Score:", round(f1, 4))

print("\nGenerated files:")
print("- model_leaderboard.csv")
print("- tuning_results.csv")
print("- evaluation_report.md")
print("- feature_importance.csv (when supported)")
print("- models/best_churn_model.joblib")
print("- plots/confusion_matrix.png")
print("- plots/model_leaderboard.png")
print("- plots/feature_importance.png (when supported)")
