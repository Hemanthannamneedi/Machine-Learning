
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Create a folder for plots
os.makedirs("plots", exist_ok=True)

# 1. Load the dataset
df = pd.read_csv("churn.csv")

# Convert target: No = 0, Yes = 1
df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

if df["churn"].isnull().any():
    raise ValueError("Unexpected labels found in churn column.")

# Convert categorical features into numeric columns
X = pd.get_dummies(
    df.drop(columns=["churn"]),
    columns=["contract"],
    drop_first=True,
    dtype=int
)
y = df["churn"]

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 3. Train the Day 48-style Decision Tree
decision_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
decision_tree.fit(X_train, y_train)
dt_predictions = decision_tree.predict(X_test)

# 4. Train the Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
random_forest.fit(X_train, y_train)
rf_predictions = random_forest.predict(X_test)

# 5. Evaluate classification metrics
def evaluate_model(name, y_true, predictions):
    print("\n---", name, "---")
    print("Accuracy :", round(accuracy_score(y_true, predictions), 3))
    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 3
    ))
    print("Recall   :", round(
        recall_score(y_true, predictions, zero_division=0), 3
    ))
    print("F1-score :", round(
        f1_score(y_true, predictions, zero_division=0), 3
    ))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Stay", "Churn"],
        zero_division=0
    ))

evaluate_model("Decision Tree (max_depth=3)", y_test, dt_predictions)
evaluate_model("Random Forest (100 trees)", y_test, rf_predictions)

# 6. Plot Random Forest confusion matrix
cm = confusion_matrix(y_test, rf_predictions, labels=[0, 1])

print("\nRandom Forest confusion matrix:")
print(cm)
print("Rows = actual labels; columns = predicted labels")
print("Label order: Stay (0), Churn (1)")

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Churn"]
)
display.plot(values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

# 7. Extract and plot top 5 feature importances
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 5 Important Features:")
print(importance.head(5).to_string(index=False))

top5 = importance.head(5).sort_values("Importance")

plt.figure(figsize=(9, 5))
plt.barh(top5["Feature"], top5["Importance"])
plt.xlabel("Feature Importance")
plt.title("Top 5 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()

# 8. Tune n_estimators
estimator_results = []

for n in [50, 100, 200]:
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    estimator_results.append({
        "n_estimators": n,
        "accuracy": accuracy
    })

results = pd.DataFrame(estimator_results)

print("\n--- n_estimators Experiment ---")
print(results.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(
    results["n_estimators"],
    results["accuracy"],
    marker="o"
)
plt.xticks([50, 100, 200])
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Test Accuracy")
plt.title("Random Forest Accuracy vs Number of Trees")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/estimator_accuracy.png", dpi=150)
plt.close()

# 9. Compare Decision Tree and Random Forest
comparison = pd.DataFrame({
    "Model": ["Decision Tree (depth=3)", "Random Forest (100 trees)"],
    "Accuracy": [
        accuracy_score(y_test, dt_predictions),
        accuracy_score(y_test, rf_predictions)
    ],
    "Precision": [
        precision_score(y_test, dt_predictions, zero_division=0),
        precision_score(y_test, rf_predictions, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, dt_predictions, zero_division=0),
        recall_score(y_test, rf_predictions, zero_division=0)
    ],
    "F1-score": [
        f1_score(y_test, dt_predictions, zero_division=0),
        f1_score(y_test, rf_predictions, zero_division=0)
    ]
})

print("\n--- Model Comparison ---")
print(comparison.to_string(index=False))

best_index = comparison["F1-score"].idxmax()
print("\nBest model by test F1-score:", comparison.loc[best_index, "Model"])

print("\nPlots saved in the plots folder.")
print("Day 49 completed.")
