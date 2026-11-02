
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_FILE = "churn.csv"
os.makedirs("plots", exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("Dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Convert target labels into numeric values
df["churn"] = (
    df["churn"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"no": 0, "yes": 1})
)

df = df.dropna(subset=["churn"]).copy()
df["churn"] = df["churn"].astype(int)

X = df.drop(columns=["churn"])
y = df["churn"]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nClass distribution:")
print(y.value_counts().sort_index())

minority_count = y.value_counts().min()
majority_count = y.value_counts().max()

print("\nMinority class count:", minority_count)
print("Majority class count:", majority_count)

# --------------------------------------------------
# 2. Split the dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

# --------------------------------------------------
# 3. Preprocessing pipeline
# --------------------------------------------------

numeric_pipeline = SklearnPipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = SklearnPipeline(
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

# --------------------------------------------------
# 4. Define three approaches
# --------------------------------------------------

models = {
    "Baseline Random Forest": ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    ),

    "Class Weight Balanced": ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )
}

# SMOTE pipeline
# SMOTE is applied only during model.fit(), not during prediction.
train_counts = y_train.value_counts()

if len(train_counts) < 2 or train_counts.min() < 2:
    raise ValueError(
        "SMOTE needs at least 2 training examples in each class. "
        "Use a larger dataset or a different train/test split."
    )

smote_neighbors = min(5, int(train_counts.min()) - 1)

models["SMOTE + Random Forest"] = ImbPipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "smote",
            SMOTE(
                random_state=42,
                k_neighbors=smote_neighbors
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# --------------------------------------------------
# 5. Train and evaluate all approaches
# --------------------------------------------------

results = []

for model_name, pipeline in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)

    start_time = time.perf_counter()

    pipeline.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time": training_time
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

# --------------------------------------------------
# 6. Compare results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest model based on F1 Score:")
print(best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))

# --------------------------------------------------
# 7. Plot comparison
# --------------------------------------------------

metrics = ["Precision", "Recall", "F1 Score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Class Imbalance Handling - Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "plots/imbalance_comparison.png"
)

plt.show()

print("\nComparison chart saved in plots folder.")
print("Results saved in model_comparison.csv")
