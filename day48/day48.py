
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("churn.csv")

print("First five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Map target labels: No = 0, Yes = 1
df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

if df["churn"].isnull().any():
    raise ValueError("Unexpected churn labels found in the dataset.")

# 3. Convert categorical features into numeric columns
X = pd.get_dummies(
    df.drop(columns=["churn"]),
    columns=["contract"],
    drop_first=True,
    dtype=int
)
y = df["churn"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Train Logistic Regression
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\n--- Logistic Regression ---")
print("Predictions:", logistic_predictions)
print("Actual values:", y_test.to_list())
print("Accuracy:", round(logistic_accuracy, 3))
print(classification_report(
    y_test,
    logistic_predictions,
    labels=[0, 1],
    target_names=["Stay", "Churn"],
    zero_division=0
))

# 6. Train a Decision Tree with depth limited to 3
tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)

print("\n--- Decision Tree (max_depth=3) ---")
print("Predictions:", tree_predictions)
print("Actual values:", y_test.to_list())
print("Accuracy:", round(tree_accuracy, 3))
print(classification_report(
    y_test,
    tree_predictions,
    labels=[0, 1],
    target_names=["Stay", "Churn"],
    zero_division=0
))

# 7. Visualize the Decision Tree
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(18, 9))
plot_tree(
    tree_model,
    feature_names=list(X.columns),
    class_names=["Stay", "Churn"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Customer Churn Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig("plots/decision_tree.png", dpi=150)
plt.close()

print("\nTree visualization saved to plots/decision_tree.png")

# 8. Compare the two models
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree (depth=3)"],
    "Test Accuracy": [logistic_accuracy, tree_accuracy]
})

print("\n--- Model Comparison ---")
print(comparison.to_string(index=False))

# 9. Experiment with Decision Tree depth
print("\n--- Overfitting Experiment ---")

depth_results = []

for depth in [3, 5, 10]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, model.predict(X_train)
    )
    test_accuracy = accuracy_score(
        y_test, model.predict(X_test)
    )

    depth_results.append({
        "max_depth": depth,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

depth_df = pd.DataFrame(depth_results)
print(depth_df.to_string(index=False))

print("\nDay 48 completed.")
