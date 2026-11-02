
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create output folders
os.makedirs("plots", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("customers.csv")

print("--- Original Dataset ---")
print(df.head())
print("\nOriginal shape:", df.shape)

# 2. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing numeric values with median
numeric_columns = [
    "Age", "Income", "MonthlyCharges",
    "Tenure", "SupportCalls"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with mode
categorical_columns = ["Contract", "Churn"]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# 3. Create feature: TotalCharges
df["TotalCharges"] = df["MonthlyCharges"] * df["Tenure"]

# 4. Create feature: AgeGroup
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 45, 60, float("inf")],
    labels=["Young", "Adult", "Middle-aged", "Senior"],
    right=True
)

# 5. Create feature: IncomeCategory
df["IncomeCategory"] = pd.cut(
    df["Income"],
    bins=[0, 30000, 60000, 90000, float("inf")],
    labels=["Low", "Medium", "High", "Very High"],
    right=True
)

# 6. Additional feature: TenureGroup
df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 12, 36, float("inf")],
    labels=["New Customer", "Regular Customer", "Loyal Customer"]
)

# 7. Additional feature: SupportCallRate
df["SupportCallRate"] = (
    df["SupportCalls"] / (df["Tenure"] + 1)
)

# 8. Detect outliers using the IQR method
outlier_columns = [
    "Age", "Income", "MonthlyCharges", "TotalCharges"
]

for column in outlier_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    df[column + "_Outlier"] = (
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    )

    print("\n", column, "outlier count:",
          int(df[column + "_Outlier"].sum()))

# 9. Display engineered dataset
print("\n--- Engineered Dataset ---")
print(df.head())

print("\nNew dataset shape:", df.shape)

new_features = [
    "TotalCharges",
    "AgeGroup",
    "IncomeCategory",
    "TenureGroup",
    "SupportCallRate"
]

print("\n--- New Features ---")
print(df[new_features].head())

# 10. Feature distribution charts
df["AgeGroup"].value_counts().sort_index().plot(
    kind="bar",
    title="Customer Distribution by Age Group"
)
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("plots/age_group.png", dpi=150)
plt.close()

df["IncomeCategory"].value_counts().sort_index().plot(
    kind="bar",
    title="Customer Distribution by Income Category"
)
plt.xlabel("Income Category")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("plots/income_category.png", dpi=150)
plt.close()

# 11. Visualize Total Charges against Tenure
plt.figure(figsize=(8, 5))
plt.scatter(df["Tenure"], df["TotalCharges"])
plt.xlabel("Tenure (Months)")
plt.ylabel("Total Charges")
plt.title("Total Charges vs Tenure")
plt.tight_layout()
plt.savefig("plots/total_charges.png", dpi=150)
plt.close()

# 12. Save the engineered dataset
df.to_csv("engineered_customers.csv", index=False)

print("\nEngineered dataset saved as engineered_customers.csv")
print("All plots saved in the plots folder.")
print("\nDay 51 completed successfully.")
