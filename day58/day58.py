
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reproducibility
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Generate numerical sequences
def generate_sequences(samples):
    starts = np.random.uniform(-10, 10, samples)
    steps = np.random.uniform(0.1, 1.0, samples)

    sequence_data = np.array([
        start + step * np.arange(11)
        for start, step in zip(starts, steps)
    ], dtype=np.float32)

    # First 10 values are inputs; 11th value is target
    X = sequence_data[:, :10]
    y = sequence_data[:, 10]

    X = X.reshape(samples, 10, 1)

    return X, y


X_train, y_train = generate_sequences(5000)
X_test, y_test = generate_sequences(1000)

print("Training input shape:", X_train.shape)
print("Training target shape:", y_train.shape)
print("Testing input shape:", X_test.shape)

# 2. Build RNN model
def build_rnn(hidden_units):
    model = models.Sequential([
        layers.Input(shape=(10, 1)),
        layers.SimpleRNN(
            hidden_units,
            activation="tanh"
        ),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# 3. Train and compare different hidden unit sizes
hidden_sizes = [8, 16, 32]
results = []
histories = {}
trained_models = {}

for units in hidden_sizes:
    print(f"\nTraining RNN with {units} hidden units")

    tf.keras.utils.set_random_seed(42)

    model = build_rnn(units)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate on test data
    test_loss, test_mae = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(X_test, verbose=0).flatten()

    test_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    results.append({
        "Hidden Units": units,
        "Test Loss (MSE)": test_loss,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Training Time (seconds)": training_time,
        "Epochs Trained": len(history.history["loss"])
    })

    histories[units] = history
    trained_models[units] = model

    model.save(f"models/rnn_{units}_units.keras")

    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

# 4. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("rnn_comparison.csv", index=False)

print("\n========== RNN COMPARISON ==========")
print(results_df.to_string(index=False))

# 5. Select model with lowest test MAE for demonstration
best_index = results_df["Test MAE"].idxmin()
best_units = int(results_df.loc[best_index, "Hidden Units"])
best_model = trained_models[best_units]

print(f"\nLowest test MAE in this comparison: {best_units} hidden units")

# 6. Predict next number from a sample sequence
sample_sequence = np.array(
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    dtype=np.float32
).reshape(1, 10, 1)

actual_next = 20.0
predicted_next = float(
    best_model.predict(sample_sequence, verbose=0)[0][0]
)

print("\n========== SEQUENCE PREDICTION ==========")
print("Input sequence:", sample_sequence.flatten())
print("Actual next number:", actual_next)
print(f"Predicted next number: {predicted_next:.4f}")

# Save prediction details
prediction_df = pd.DataFrame([{
    "Input Sequence": str(sample_sequence.flatten().tolist()),
    "Actual Next Number": actual_next,
    "Predicted Next Number": predicted_next,
    "Absolute Error": abs(actual_next - predicted_next),
    "Selected Hidden Units": best_units
}])

prediction_df.to_csv("sample_prediction.csv", index=False)

# 7. Plot training and validation loss
plt.figure(figsize=(9, 5))

for units in hidden_sizes:
    history = histories[units]
    plt.plot(
        history.history["loss"],
        label=f"{units} Units - Training"
    )
    plt.plot(
        history.history["val_loss"],
        linestyle="--",
        label=f"{units} Units - Validation"
    )

plt.title("RNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/training_loss_comparison.png")
plt.close()

# 8. Compare test MAE
plt.figure(figsize=(8, 5))
plt.bar(
    results_df["Hidden Units"].astype(str),
    results_df["Test MAE"]
)
plt.title("RNN Hidden Unit Comparison")
plt.xlabel("Number of Hidden Units")
plt.ylabel("Test MAE")
plt.tight_layout()
plt.savefig("plots/hidden_unit_comparison.png")
plt.close()

# 9. Visualize actual vs predicted values
test_predictions = best_model.predict(X_test[:50], verbose=0).flatten()

plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label="Actual Values", marker="o")
plt.plot(test_predictions, label="Predicted Values", marker="x")
plt.title(f"RNN Predictions ({best_units} Hidden Units)")
plt.xlabel("Sample")
plt.ylabel("Next Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/prediction_visualization.png")
plt.close()

# 10. Save evaluation report
with open("evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write("RNN Sequence Prediction Evaluation\n")
    file.write("==================================\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nSample Prediction\n")
    file.write(f"Actual next number: {actual_next}\n")
    file.write(f"Predicted next number: {predicted_next:.4f}\n")
    file.write(f"Selected hidden units: {best_units}\n")

print("\nProject completed successfully!")
print("Models saved in models/")
print("Comparison saved in rnn_comparison.csv")
print("Visualizations saved in plots/")
