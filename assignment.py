import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# Load the Iris dataset from seaborn's built-in datasets
try:
    iris_df = sns.load_dataset("iris")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows of the dataset
print("\nPreview of the dataset:")
print(iris_df.head())

# Explore the structure of the dataset
print("\nDataset Information:")
iris_df.info()

# Check for missing values
missing_values = iris_df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Handle missing values if any
if missing_values.any():
    iris_df = iris_df.dropna()
    print("\nMissing values were found and have been removed.")
else:
    print("\nNo missing values found.")

# Task 2: Basic Data Analysis
# Display basic statistics for numerical columns
print("\nBasic Statistics:")
print(iris_df.describe())

# Calculate mean values grouped by species
species_mean = iris_df.groupby('species').mean()
print("\nAverage values by species:")
print(species_mean)

# Identify insights
print("\nInsights:")
print("Versicolor and Virginica species have higher average petal lengths compared to Setosa.")

# Task 3: Data Visualization
sns.set(style="whitegrid")

# 1. Line chart: Petal Length Trend by Species
print("\nCreating a line chart...")
plt.figure(figsize=(8, 6))
sns.lineplot(data=iris_df, x=iris_df.index, y="petal_length", hue="species", marker="o")
plt.title("Petal Length Trend by Species")
plt.xlabel("Index")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.savefig("line_chart.png")
plt.close()
print("Line chart saved as 'line_chart.png'.")

# 2. Bar chart: Average Petal Length by Species
print("\nCreating a bar chart...")
plt.figure(figsize=(8, 6))
sns.barplot(x=species_mean.index, y=species_mean["petal_length"], palette="viridis")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.savefig("bar_chart.png")
plt.close()
print("Bar chart saved as 'bar_chart.png'.")

# 3. Histogram: Distribution of Petal Lengths
print("\nCreating a histogram...")
plt.figure(figsize=(8, 6))
sns.histplot(data=iris_df, x="petal_length", kde=True, bins=10, color="blue")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.close()
print("Histogram saved as 'histogram.png'.")

# 4. Scatter plot: Petal Length vs Petal Width
print("\nCreating a scatter plot...")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_df, x="petal_length", y="petal_width", hue="species", style="species", palette="Set1")
plt.title("Petal Length vs Petal Width by Species")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(title="Species")
plt.savefig("scatter_plot.png")
plt.close()
print("Scatter plot saved as 'scatter_plot.png'.")

# Additional Notes:
# This script performs basic exploratory data analysis and saves visualizations as image files.
# Replace the dataset with your own CSV or other data source if needed.
