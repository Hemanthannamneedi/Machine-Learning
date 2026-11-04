
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("TensorFlow version:", tf.__version__)

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# 2. Prepare image data
# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Prepared training shape:", x_train.shape)
print("Prepared testing shape:", x_test.shape)

# 3. Build CNN architecture
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # First convolution block
    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into a one-dimensional vector
    layers.Flatten(),

    # Classification layers
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Architecture:")
model.summary()

# 5. Train the model
start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

training_time = time.time() - start_time

# 6. Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n========== TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Training Time: {training_time:.2f} seconds")

# 7. Generate predictions
probabilities = model.predict(x_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# 8. Save trained model
model.save("models/mnist_cnn.keras")

# 9. Save test results
with open("evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write("MNIST CNN Evaluation Report\n")
    file.write("===========================\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    file.write(f"Training Time: {training_time:.2f} seconds\n")
    file.write("\nClassification Report:\n")
    file.write(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

# 10. Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/accuracy_curve.png")
plt.close()

# 11. Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/loss_curve.png")
plt.close()

# 12. Visualize sample predictions
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap="gray")

    predicted_label = predictions[i]
    actual_label = y_test[i]

    plt.title(
        f"Actual: {actual_label}\nPredicted: {predicted_label}"
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("plots/sample_predictions.png")
plt.close()

# 13. Confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("MNIST CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()

for row in range(10):
    for col in range(10):
        plt.text(
            col, row, str(cm[row, col]),
            ha="center", va="center", fontsize=8
        )

plt.xticks(range(10))
plt.yticks(range(10))
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

print("\nProject completed successfully!")
print("Saved model: models/mnist_cnn.keras")
print("Saved report: evaluation_report.txt")
print("Saved visualizations in: plots/")
