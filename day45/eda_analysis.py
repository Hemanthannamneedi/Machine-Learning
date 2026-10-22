
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a folder for chart images
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# PHASE 1: LOAD DATA
df = pd.read_csv("students.csv")

print("=== ORIGINAL DATASET ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn data types:\n", df.dtypes)
print("\nStatistical summary:\n", df.describe())

# Simulate common data-quality problems for practice:
# one missing mark, one missing city, and one duplicate row
df.loc[1, "marks"] = float("nan")
df.loc[3, "city"] = None
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

print("\n=== DATA QUALITY CHECK ===")
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# PHASE 2: CLEAN DATA
# Standardize text columns
for column in ["name", "gender", "city", "course"]:
    df[column] = df[column].astype("string").str.strip()

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Fill missing numeric values with the median
df["marks"] = df["marks"].fillna(df["marks"].median())

# Fill missing city values with the most frequent city
df["city"] = df["city"].fillna(df["city"].mode()[0])

print("\n=== CLEANED DATASET ===")
print(df)
print("\nRemaining missing values:\n", df.isnull().sum())
print("Cleaned dataset shape:", df.shape)

# PHASE 3: VISUALIZE PATTERNS

# 1. Histogram of marks
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="marks", bins=5, kde=True)
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("plots/marks_histogram.png")
plt.close()

# 2. Scatter plot: study hours vs marks
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="study_hours", y="marks",
    hue="course", s=100
)
plt.title("Study Hours vs Student Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("plots/study_hours_vs_marks.png")
plt.close()

# 3. Box plot: marks by course
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="course", y="marks")
plt.title("Marks Distribution by Course")
plt.xlabel("Course")
plt.ylabel("Marks")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("plots/marks_by_course.png")
plt.close()

# 4. Bar chart: average marks by course
course_average = df.groupby("course")["marks"].mean().sort_values(
    ascending=False
)

plt.figure(figsize=(8, 5))
sns.barplot(x=course_average.index, y=course_average.values)
plt.title("Average Marks by Course")
plt.xlabel("Course")
plt.ylabel("Average Marks")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("plots/average_marks_by_course.png")
plt.close()

# 5. Bar chart: average marks by study hours
hours_average = df.groupby("study_hours")["marks"].mean().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(x=hours_average.index, y=hours_average.values)
plt.title("Average Marks by Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Average Marks")
plt.tight_layout()
plt.savefig("plots/average_marks_by_study_hours.png")
plt.close()

# PHASE 4: SUMMARIZE INSIGHTS
print("\n=== EDA INSIGHTS ===")
print("1. Average marks:", round(df["marks"].mean(), 2))
print("2. Median marks:", round(df["marks"].median(), 2))
print(
    "3. Highest average marks by course:",
    course_average.index[0],
    "(" + str(round(course_average.iloc[0], 2)) + ")"
)
print(
    "4. Study hours and marks correlation:",
    round(df["study_hours"].corr(df["marks"]), 2)
)
print(
    "5. Highest average marks by study hours:",
    int(hours_average.idxmax()),
    "hours (average:",
    round(hours_average.max(), 2),
    ")"
)

print("\nAll EDA charts saved in the plots folder.")
