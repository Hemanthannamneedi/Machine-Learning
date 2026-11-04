
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==================================================
# 1. REPRODUCIBILITY AND FOLDERS
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv("churn.csv")

df.columns = df.columns.str.strip().str.lower()

print("Dataset preview:")
print(df.head())

print("\nDataset shape:", df.shape)


# Convert churn labels to 0 and 1
if df["churn"].dtype == "object":
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


# ==================================================
# 3. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)


# ==================================================
# 4. DATA PREPROCESSING
# ==================================================

numeric_features = X.select_dtypes(
    include=["number"]
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

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to dense arrays for Keras
X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

y_train_array = y_train.to_numpy(dtype=np.float32)
y_test_array = y_test.to_numpy(dtype=np.float32)

input_features = X_train_processed.shape[1]

print("\nNumber of input features:", input_features)


# ==================================================
# 5. ANN MODEL CREATION
# ==================================================

def build_ann(number_of_neurons):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_features,)),

        tf.keras.layers.Dense(
            number_of_neurons,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==================================================
# 6. TRAIN AND COMPARE ANN ARCHITECTURES
# ==================================================

neuron_counts = [10, 5, 20, 50]

results = []
trained_models = {}
training_histories = {}

for neurons in neuron_counts:

    print("\n" + "=" * 45)
    print("Training ANN with", neurons, "hidden neurons")

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_ann(neurons)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    start_time = time.perf_counter()

    history = model.fit(
        X_train_processed,
        y_train_array,
        epochs=100,
        batch_size=4,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    training_time = time.perf_counter() - start_time

    loss, accuracy = model.evaluate(
        X_test_processed,
        y_test_array,
        verbose=0
    )

    probabilities = model.predict(
        X_test_processed,
        verbose=0
    ).flatten()

    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(
        y_test_array,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test_array,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test_array,
        predictions,
        zero_division=0
    )

    results.append({
        "Hidden Neurons": neurons,
        "Accuracy": accuracy_score(
            y_test_array, predictions
        ),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Test Loss": loss,
        "Training Time": training_time,
        "Epochs": len(history.history["loss"])
    })

    trained_models[neurons] = model
    training_histories[neurons] = history

    print("Test Loss:", round(loss, 4))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Epochs:", len(history.history["loss"]))


# ==================================================
# 7. SAVE THE 10-NEURON MODEL
# ==================================================

trained_models[10].save(
    "models/ann_10_neurons.keras"
)

print("\n10-neuron ANN saved successfully.")


# ==================================================
# 8. SAVE MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nANN ARCHITECTURE COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "ann_comparison.csv",
    index=False
)


# ==================================================
# 9. PLOT TRAINING LOSS FOR 10 NEURONS
# ==================================================

history = training_histories[10]

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training and Validation Loss (10 Neurons)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig("plots/ann_training_loss.png")
plt.close()


# ==================================================
# 10. COMPARE DIFFERENT NEURON COUNTS
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Hidden Neurons"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["Hidden Neurons"],
    results_df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.title("ANN Performance vs Hidden Neurons")
plt.xlabel("Number of Hidden Neurons")
plt.ylabel("Score")
plt.xticks(neuron_counts)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig("plots/neuron_comparison.png")
plt.close()


# ==================================================
# 11. FINAL 10-NEURON MODEL PREDICTION
# ==================================================

final_model = trained_models[10]

final_predictions = (
    final_model.predict(
        X_test_processed,
        verbose=0
    ).flatten() >= 0.5
).astype(int)

print("\n10-NEURON ANN CLASSIFICATION REPORT")
print(
    classification_report(
        y_test_array,
        final_predictions,
        zero_division=0
    )
)


# ==================================================
# 12. SAVE PREPROCESSOR
# ==================================================

import joblib

joblib.dump(
    preprocessor,
    "models/preprocessor.joblib"
)


# ==================================================
# 13. FINAL SUMMARY
# ==================================================

best_result = results_df.iloc[0]

print("\nPROJECT COMPLETED SUCCESSFULLY")

print(
    "Best architecture:",
    int(best_result["Hidden Neurons"]),
    "hidden neurons"
)

print("Best F1 Score:", round(best_result["F1 Score"], 4))

print("\nGenerated files:")
print("- ann_comparison.csv")
print("- models/ann_10_neurons.keras")
print("- models/preprocessor.joblib")
print("- plots/ann_training_loss.png")
print("- plots/neuron_comparison.png")
