
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Create plots folder
os.makedirs("plots", exist_ok=True)

# 2. Load dataset
df = pd.read_csv("churn.csv")

print("--- Dataset Preview ---")
print(df.head())
print("\nDataset shape:", df.shape)

# 3. Prepare target labels
df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

if df["churn"].isnull().any():
    raise ValueError("Unexpected values found in churn column.")

# Convert categorical features into numerical features
X = pd.get_dummies(
    df.drop(columns=["churn"]),
    columns=["contract"],
    drop_first=True,
    dtype=int
)
y = df["churn"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Demonstrate StandardScaler
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

print("\n--- StandardScaler ---")
print("Training data mean (approximately 0):")
print(np.round(X_train_standard.mean(axis=0), 2))

# 6. Demonstrate MinMaxScaler
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

print("\n--- MinMaxScaler ---")
print("Training data minimum:")
print(np.round(X_train_minmax.min(axis=0), 2))
print("Training data maximum:")
print(np.round(X_train_minmax.max(axis=0), 2))

# 7. KNN: Compare different k values
k_values = [1, 5, 10, 20]

knn_results = []

for k in k_values:

    # Without scaling
    knn_unscaled = KNeighborsClassifier(n_neighbors=k)
    knn_unscaled.fit(X_train, y_train)
    unscaled_predictions = knn_unscaled.predict(X_test)

    unscaled_accuracy = accuracy_score(
        y_test, unscaled_predictions
    )

    # With StandardScaler
    knn_scaled = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )
    knn_scaled.fit(X_train, y_train)
    scaled_predictions = knn_scaled.predict(X_test)

    scaled_accuracy = accuracy_score(
        y_test, scaled_predictions
    )

    knn_results.append({
        "k": k,
        "Without Scaling": unscaled_accuracy,
        "With StandardScaler": scaled_accuracy
    })

knn_df = pd.DataFrame(knn_results)

print("\n--- KNN Scaling Comparison ---")
print(knn_df.to_string(index=False))

# Plot KNN accuracy comparison
plt.figure(figsize=(8, 5))
plt.plot(
    knn_df["k"],
    knn_df["Without Scaling"],
    marker="o",
    label="Without Scaling"
)
plt.plot(
    knn_df["k"],
    knn_df["With StandardScaler"],
    marker="s",
    label="With StandardScaler"
)
plt.xticks(k_values)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Test Accuracy")
plt.title("KNN: With vs Without Feature Scaling")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/knn_scaling_comparison.png", dpi=150)
plt.close()

# 8. Compare Linear SVM and RBF SVM
svm_kernels = ["linear", "rbf"]

print("\n--- SVM Kernel Comparison ---")

for kernel in svm_kernels:
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel)
    )

    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    print(
        kernel,
        "SVM accuracy:",
        round(accuracy_score(y_test, svm_predictions), 3)
    )

# 9. Build five classification models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "SVM (RBF)": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf")
    )
}

# 10. Train and evaluate all five models
comparison_results = []

for name, model in models.items():

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    predictions = model.predict(X_test)

    comparison_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Training Time (seconds)": training_time
    })

# 11. Display comparison table
comparison_df = pd.DataFrame(comparison_results)

print("\n--- Five Model Comparison ---")
print(comparison_df.round(4).to_string(index=False))

# Save comparison results
comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)

# 12. Identify the best model by F1-score
best_model = comparison_df.loc[
    comparison_df["F1-score"].idxmax()
]

print("\nBest model based on F1-score:")
print(best_model.to_string())

# 13. Plot model accuracy
plt.figure(figsize=(9, 5))
plt.bar(
    comparison_df["Model"],
    comparison_df["Accuracy"]
)
plt.xlabel("Classification Models")
plt.ylabel("Test Accuracy")
plt.title("Comparison of Five Classification Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("plots/model_accuracy_comparison.png", dpi=150)
plt.close()

print("\nPlots saved in the plots folder.")
print("Model comparison saved as model_comparison.csv")
print("\nDay 52 completed successfully.")
