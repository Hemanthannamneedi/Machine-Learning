
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Load data
df = pd.read_csv("house_prices.csv")

print("--- Dataset Preview ---")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Basic cleaning
df = df.drop_duplicates()
df = df.dropna(subset=["price_lakhs"])

# Remove rows with invalid target values
df = df[df["price_lakhs"] > 0]

if len(df) < 5:
    raise ValueError("Dataset needs at least 5 valid rows.")

# 3. Separate features and target
X = df.drop(columns=["price_lakhs"])
y = df["price_lakhs"]

numeric_features = ["size_sqft", "bedrooms", "age_years"]
categorical_features = ["location"]

# 4. Prepare preprocessing steps
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Build a complete pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

print("\nModel training completed.")

# 8. Evaluate on unseen test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print("MAE :", round(mae, 2), "lakhs")
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2), "lakhs")
print("R2  :", round(r2, 3))

# 9. Perform 5-fold cross-validation on training data
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

cv_rmse = np.sqrt(-cv_scores)

print("\n--- 5-Fold Cross-Validation ---")
print("RMSE for each fold:", np.round(cv_rmse, 2))
print("Mean CV RMSE:", round(cv_rmse.mean(), 2), "lakhs")

# 10. Plot actual vs predicted prices
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual vs Predicted House Prices")

# Reference line for easier comparison
lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())
plt.plot([lower, upper], [lower, upper], linestyle="--")

plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=150)
plt.close()

# 11. Save the complete pipeline
model_path = "models/house_model.joblib"
joblib.dump(model, model_path)

print("\nSaved model to:", model_path)

# 12. Load the saved model
loaded_model = joblib.load(model_path)

# 13. Predict the price of a new house
new_house = pd.DataFrame([{
    "size_sqft": 1700,
    "bedrooms": 3,
    "age_years": 5,
    "location": "City"
}])

predicted_price = loaded_model.predict(new_house)[0]

print("\n--- New House Prediction ---")
print("House details:")
print(new_house.to_string(index=False))
print("Predicted price:", round(predicted_price, 2), "lakhs")

print("\nDay 50 ML pipeline completed successfully.")
