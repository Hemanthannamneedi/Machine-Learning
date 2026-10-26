
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("house_prices.csv")

print("=== Dataset Preview ===")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Separate features and target
X = df[["size_sqft", "bedrooms", "age_years"]]
y = df["price_lakhs"]

# 3. Train-test split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Single Train-Test Split ===")
print("Actual prices:", y_test.tolist())
print("Predicted prices:", np.round(y_pred, 2).tolist())
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

# 7. Five-fold cross-validation: R²
cv5_r2 = cross_val_score(
    LinearRegression(), X, y, cv=5, scoring="r2"
)

print("\n=== 5-Fold Cross-Validation ===")
print("R² scores:", np.round(cv5_r2, 4))
print("Mean R²:", round(cv5_r2.mean(), 4))
print("Standard deviation:", round(cv5_r2.std(), 4))

# 8. Compare 5-fold and 10-fold using MSE
# Negative MSE is returned by sklearn, so convert it to positive MSE.
for k in [5, 10]:
    cv_mse = -cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=k,
        scoring="neg_mean_squared_error"
    )

    cv_rmse = np.sqrt(cv_mse)

    print(f"\n=== {k}-Fold Cross-Validation: Error Metrics ===")
    print("MSE scores:", np.round(cv_mse, 2))
    print("Mean MSE:", round(cv_mse.mean(), 2))
    print("Mean RMSE:", round(cv_rmse.mean(), 2))
    print("MSE standard deviation:", round(cv_mse.std(), 2))

# 9. Check whether 10-fold R² is suitable for this dataset
smallest_test_fold = len(df) // 10

print("\n=== 10-Fold R² Check ===")
if smallest_test_fold < 2:
    print(
        "10-fold R² is unreliable for this small dataset because "
        "some test folds contain only one sample."
    )
    print("Use a larger dataset for a fair 10-fold R² comparison.")
else:
    cv10_r2 = cross_val_score(
        LinearRegression(), X, y, cv=10, scoring="r2"
    )
    print("R² scores:", np.round(cv10_r2, 4))
    print("Mean R²:", round(cv10_r2.mean(), 4))
    print("Standard deviation:", round(cv10_r2.std(), 4))
