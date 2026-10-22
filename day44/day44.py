
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("students.csv")

# Set a clean visualization style
sns.set_theme(style="whitegrid")

print("=== Dataset Preview ===")
print(df.head())

print("\n=== Statistical Summary ===")
print(df[["age", "study_hours", "marks"]].describe())

# 1. Histogram: Distribution of student marks
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="marks", bins=5, kde=True)
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("01_marks_histogram.png")
plt.show()

# 2. Scatter Plot: Study hours vs marks
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="course",
    s=100
)
plt.title("Study Hours vs Student Marks")
plt.xlabel("Study Hours per Day")
plt.ylabel("Marks")
plt.legend(title="Course")
plt.tight_layout()
plt.savefig("02_study_hours_scatter.png")
plt.show()

# 3. Box Plot: Marks by course
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="course", y="marks")
plt.title("Marks Distribution by Course")
plt.xlabel("Course")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("03_marks_boxplot.png")
plt.show()

# 4. Bar Chart: Average marks by course
average_marks = df.groupby("course", as_index=False)["marks"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(data=average_marks, x="course", y="marks")
plt.title("Average Marks by Course")
plt.xlabel("Course")
plt.ylabel("Average Marks")
plt.tight_layout()
plt.savefig("04_average_marks_bar.png")
plt.show()

# 5. Line Plot: Marks by study hours
sorted_df = df.sort_values("study_hours")

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=sorted_df,
    x="study_hours",
    y="marks",
    marker="o"
)
plt.title("Study Hours and Marks Trend")
plt.xlabel("Study Hours per Day")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("05_study_hours_line.png")
plt.show()

print("\nAll five charts generated and saved successfully.")
