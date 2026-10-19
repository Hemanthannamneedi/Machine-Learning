
import pandas as pd

# 1. Load the CSV dataset
df = pd.read_csv("students.csv")

print("=== First 5 Rows ===")
print(df.head())

print("\n=== Last 5 Rows ===")
print(df.tail())

print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns)

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Dataset Information ===")
df.info()

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Course Counts ===")
print(df["course"].value_counts())

print("\n=== Students with Marks Above 80 ===")
print(df[df["marks"] > 80])

print("\n=== Students from Hyderabad ===")
print(df[df["city"] == "Hyderabad"])
