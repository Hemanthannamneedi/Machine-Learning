import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_csv("students_dirty.csv")

print("=== Original Dataset ===")
print(df)

# 2. Identify missing values
print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())

# 3. Compare dropna() and fillna()
dropped_df = df.dropna()

print("\n=== After dropna() ===")
print(dropped_df)
print("Rows remaining:", len(dropped_df))
print("Rows removed:", len(df) - len(dropped_df))

filled_df = df.copy()

# Fill numerical missing values with the median
filled_df["age"] = filled_df["age"].fillna(
    filled_df["age"].median()
)
filled_df["marks"] = filled_df["marks"].fillna(
    filled_df["marks"].median()
)

# Fill missing city values with the most frequent city
filled_df["city"] = filled_df["city"].fillna(
    filled_df["city"].mode()[0]
)

print("\n=== After fillna() ===")
print(filled_df)
print("Rows remaining:", len(filled_df))

# 4. Encode ordinal data: Education
# Required order: High School < Bachelors < Masters
education_encoder = LabelEncoder()

# Set the intended ordinal order explicitly.
education_encoder.classes_ = np.array([
    "High School", "Bachelors", "Masters"
])

filled_df["education_encoded"] = education_encoder.transform(
    filled_df["education"]
)

print("\n=== Education Label Encoding ===")
print(filled_df[["education", "education_encoded"]])

# 5. One-hot encode nominal columns
encoded_df = pd.get_dummies(
    filled_df,
    columns=["gender", "city"],
    drop_first=True,
    dtype=int
)

# 6. Remove original text columns.
# Name is an identifier, not a useful numeric model feature.
encoded_df = encoded_df.drop(columns=["name", "education"])

print("\n=== Final Encoded Dataset ===")
print(encoded_df)

print("\n=== Final Column Data Types ===")
print(encoded_df.dtypes)

# 7. Verify no string columns remain
string_columns = encoded_df.select_dtypes(
    include=["object", "string"]
).columns

if len(string_columns) == 0:
    print("\nSuccess: No string columns remain.")
else:
    print("\nString columns still present:", list(string_columns))

# 8. Verify there are no missing values
print("\n=== Remaining Missing Values ===")
print(encoded_df.isnull().sum())
