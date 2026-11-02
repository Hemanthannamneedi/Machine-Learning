
import os
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("churn.csv")

print("Dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)

# Clean target column
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

print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Ensure there are enough minority samples for 3-fold CV
class_counts = y_train.value_counts()

if len(class_counts) < 2 or class_counts.min() < 3:
    raise ValueError(
        "At least 3 training examples per class are required "
        "for this 3-fold stratified GridSearchCV. "
        "Use a larger dataset if necessary."
    )

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# --------------------------------------------------
# 4. GridSearchCV - Gradient Boosting
# --------------------------------------------------

gb_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ]
)

gb_parameters = {
    "model__n_estimators": [50, 100],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [1, 2]
}

print("\nTuning Gradient Boosting...")

gb_grid = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=gb_parameters,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print("Gradient Boosting best parameters:")
print(gb_grid.best_params_)
print("Best CV F1:", round(gb_grid.best_score_, 4))

# --------------------------------------------------
# 5. GridSearchCV - XGBoost
# --------------------------------------------------

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            )
        )
    ]
)

xgb_parameters = {
    "model__n_estimators": [50, 100],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3]
}

print("\nTuning XGBoost...")

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_parameters,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("XGBoost best parameters:")
print(xgb_grid.best_params_)
print("Best CV F1:", round(xgb_grid.best_score_, 4))

# --------------------------------------------------
# 6. Define six assignment models + Gradient Boosting
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]
    ),

    "Decision Tree": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=3,
                random_state=42
            ))
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]
    ),

    "KNN": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=3))
        ]
    ),

    "SVM": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", SVC(kernel="rbf"))
        ]
    ),

    "Gradient Boosting (Default)": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42))
        ]
    ),

    "Gradient Boosting (Tuned)": gb_grid.best_estimator_,

    "XGBoost (Default)": Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            ))
        ]
    ),

    "XGBoost (Tuned)": xgb_grid.best_estimator_
}

# --------------------------------------------------
# 7. Train and evaluate all models
# --------------------------------------------------

results = []

for name, model in models.items():

    print("\nTraining:", name)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Training Time (seconds)": training_time
    })

    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))

# --------------------------------------------------
# 8. Final leaderboard
# --------------------------------------------------

leaderboard = pd.DataFrame(results)

leaderboard = leaderboard.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nFINAL MODEL LEADERBOARD")
print(leaderboard.to_string(index=False))

leaderboard.to_csv(
    "model_leaderboard.csv",
    index=False
)

# --------------------------------------------------
# 9. Before vs after tuning
# --------------------------------------------------

tuning_comparison = leaderboard[
    leaderboard["Model"].isin([
        "Gradient Boosting (Default)",
        "Gradient Boosting (Tuned)",
        "XGBoost (Default)",
        "XGBoost (Tuned)"
    ])
]

print("\nDEFAULT VS TUNED COMPARISON")
print(tuning_comparison.to_string(index=False))

tuning_comparison.to_csv(
    "tuning_comparison.csv",
    index=False
)

# --------------------------------------------------
# 10. Best model
# --------------------------------------------------

winner = leaderboard.iloc[0]

print("\nBEST MODEL BASED ON TEST F1 SCORE")
print("Model:", winner["Model"])
print("Accuracy:", round(winner["Accuracy"], 4))
print("F1 Score:", round(winner["F1 Score"], 4))

# --------------------------------------------------
# 11. Leaderboard chart
# --------------------------------------------------

leaderboard.plot(
    x="Model",
    y=["Accuracy", "F1 Score"],
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Leaderboard")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig("plots/model_leaderboard.png")
plt.show()

print("\nAll results saved successfully.")
