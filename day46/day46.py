
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv("house_prices.csv")

print("=== Dataset Preview ===")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 2. Separate features and target
y = df["price_lakhs"]

# Simple Linear Regression: one feature
X_simple = df[["size_sqft"]]

# 3. Split simple model data: 80% training, 20% testing
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=11
)

# 4. Train simple linear regression
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

simple_predictions = simple_model.predict(X_test_s)

print("\n=== Simple Linear Regression ===")
print("Actual prices:", y_test_s.tolist())
print("Predicted prices:", simple_predictions.round(2).tolist())
print("MAE:", round(mean_absolute_error(y_test_s, simple_predictions), 2))
print("R² score:", round(r2_score(y_test_s, simple_predictions), 3))

# 5. Multiple Linear Regression: three features
X_multiple = df[["size_sqft", "bedrooms", "age_years"]]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multiple, y, test_size=0.2, random_state=11
)

# 6. Train multiple linear regression
multiple_model = LinearRegression()
multiple_model.fit(X_train_m, y_train_m)

multiple_predictions = multiple_model.predict(X_test_m)

print("\n=== Multiple Linear Regression ===")
print("Actual prices:", y_test_m.tolist())
print("Predicted prices:", multiple_predictions.round(2).tolist())
print("MAE:", round(mean_absolute_error(y_test_m, multiple_predictions), 2))
print("R² score:", round(r2_score(y_test_m, multiple_predictions), 3))

# 7. Compare the two models
print("\n=== Model Comparison ===")
print("Simple model MAE:", round(mean_absolute_error(y_test_s, simple_predictions), 2))
print("Multiple model MAE:", round(mean_absolute_error(y_test_m, multiple_predictions), 2))

# 8. Predict the price of a new house
new_house = pd.DataFrame({
    "size_sqft": [1700],
    "bedrooms": [3],
    "age_years": [5]
})

predicted_price = multiple_model.predict(new_house)[0]

print("\n=== New House Price Prediction ===")
print("House details:")
print(new_house)
print("Predicted price:", round(predicted_price, 2), "lakhs")
