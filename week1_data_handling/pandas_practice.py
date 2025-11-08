"""
Week 1: Pandas Fundamentals
Save as: D:\ai_engineering\week1_data_handling\pandas_practice.py

Learn data manipulation with real datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("WEEK 1 - DAY 3-4: Pandas Exercises")
print("="*60)

# ============================================
# EXERCISE 1: Creating DataFrames
# ============================================
print("\n>>> Exercise 1: DataFrame Creation")

# Method 1: From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'Salary': [70000, 80000, 75000, 65000, 90000]
}
df = pd.DataFrame(data)
print("\nDataFrame from dictionary:")
print(df)

# Method 2: From NumPy array
arr = np.random.randint(0, 100, size=(5, 3))
df_np = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print("\nDataFrame from NumPy:")
print(df_np)

# Method 3: From CSV (we'll create one first)
df.to_csv('D:/ai_engineering/datasets/sample_data.csv', index=False)
df_csv = pd.read_csv('D:/ai_engineering/datasets/sample_data.csv')
print("\nDataFrame loaded from CSV:")
print(df_csv.head())

# ============================================
# EXERCISE 2: Exploring Data
# ============================================
print("\n>>> Exercise 2: Data Exploration")

print("\nDataFrame info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

print("\nColumn names:", df.columns.tolist())
print("Shape:", df.shape)
print("Size:", df.size)
print("Data types:\n", df.dtypes)

# First/last rows
print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 2 rows:")
print(df.tail(2))

# ============================================
# EXERCISE 3: Selecting Data
# ============================================
print("\n>>> Exercise 3: Data Selection")

# Select single column
print("\nNames only:")
print(df['Name'])

# Select multiple columns
print("\nName and City:")
print(df[['Name', 'City']])

# Select rows by index
print("\nRow 2:")
print(df.iloc[2])

# Select rows by condition
print("\nPeople older than 30:")
print(df[df['Age'] > 30])

print("\nPeople from NYC:")
print(df[df['City'] == 'NYC'])

# Multiple conditions
print("\nAge > 25 AND Salary > 70000:")
print(df[(df['Age'] > 25) & (df['Salary'] > 70000)])

# ============================================
# EXERCISE 4: Data Manipulation
# ============================================
print("\n>>> Exercise 4: Data Manipulation")

# Add new column
df['Bonus'] = df['Salary'] * 0.1
print("\nWith bonus column:")
print(df)

# Modify existing column
df['Age'] = df['Age'] + 1  # Everyone gets a year older
print("\nAges increased by 1:")
print(df)

# Delete column
df_copy = df.copy()
df_copy.drop('Bonus', axis=1, inplace=True)
print("\nAfter dropping Bonus:")
print(df_copy)

# Rename columns
df_renamed = df.rename(columns={'Name': 'Employee', 'City': 'Location'})
print("\nRenamed columns:")
print(df_renamed)

# ============================================
# EXERCISE 5: Grouping and Aggregation
# ============================================
print("\n>>> Exercise 5: Grouping & Aggregation")

# Group by city
print("\nAverage salary by city:")
print(df.groupby('City')['Salary'].mean())

print("\nCount by city:")
print(df.groupby('City').size())

print("\nMultiple aggregations:")
print(df.groupby('City').agg({
    'Salary': ['mean', 'max', 'min'],
    'Age': 'mean'
}))

# ============================================
# EXERCISE 6: Sorting
# ============================================
print("\n>>> Exercise 6: Sorting")

print("\nSort by Age (ascending):")
print(df.sort_values('Age'))

print("\nSort by Salary (descending):")
print(df.sort_values('Salary', ascending=False))

print("\nSort by multiple columns:")
print(df.sort_values(['City', 'Age']))

# ============================================
# EXERCISE 7: Handling Missing Data
# ============================================
print("\n>>> Exercise 7: Missing Data")

# Create data with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, 4, 5]
})
print("\nData with missing values:")
print(df_missing)

# Check for missing values
print("\nMissing values per column:")
print(df_missing.isnull().sum())

# Drop rows with missing values
print("\nAfter dropping NaN rows:")
print(df_missing.dropna())

# Fill missing values
print("\nFill NaN with 0:")
print(df_missing.fillna(0))

print("\nFill NaN with mean:")
print(df_missing.fillna(df_missing.mean()))

# ============================================
# MINI PROJECT: Analyze Iris Dataset
# ============================================
print("\n" + "="*60)
print("MINI PROJECT: Iris Dataset Analysis")
print("="*60)

from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({
    0: 'setosa', 
    1: 'versicolor', 
    2: 'virginica'
})

print("\nFirst 5 rows:")
print(iris_df.head())

print("\nDataset info:")
print(iris_df.info())

print("\nBasic statistics:")
print(iris_df.describe())

print("\nSpecies distribution:")
print(iris_df['species_name'].value_counts())

print("\nAverage measurements by species:")
print(iris_df.groupby('species_name').mean())

# Find largest/smallest flowers
print("\nLargest sepal length:")
print(iris_df.loc[iris_df['sepal length (cm)'].idxmax()])

print("\nSmallest petal length:")
print(iris_df.loc[iris_df['petal length (cm)'].idxmin()])

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Histogram of sepal length
axes[0, 0].hist(iris_df['sepal length (cm)'], bins=20, edgecolor='black')
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Sepal Length')

# Plot 2: Scatter plot
for species in iris_df['species_name'].unique():
    data = iris_df[iris_df['species_name'] == species]
    axes[0, 1].scatter(data['sepal length (cm)'], 
                       data['sepal width (cm)'], 
                       label=species, alpha=0.6)
axes[0, 1].set_xlabel('Sepal Length (cm)')
axes[0, 1].set_ylabel('Sepal Width (cm)')
axes[0, 1].set_title('Sepal Dimensions by Species')
axes[0, 1].legend()

# Plot 3: Box plot
iris_df.boxplot(column='petal length (cm)', by='species_name', ax=axes[1, 0])
axes[1, 0].set_title('Petal Length Distribution')
axes[1, 0].set_xlabel('Species')

# Plot 4: Bar chart of averages
avg_by_species = iris_df.groupby('species_name')['sepal length (cm)'].mean()
avg_by_species.plot(kind='bar', ax=axes[1, 1], color=['red', 'green', 'blue'])
axes[1, 1].set_title('Average Sepal Length by Species')
axes[1, 1].set_ylabel('Length (cm)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week1_data_handling/iris_analysis.png', dpi=150)
print("\nVisualizations saved: iris_analysis.png")

# Save processed data
iris_df.to_csv('D:/ai_engineering/datasets/iris_processed.csv', index=False)
print("Processed data saved: iris_processed.csv")

# ============================================
# CHALLENGE EXERCISES
# ============================================
print("\n" + "="*60)
print("CHALLENGE EXERCISES")
print("="*60)

print("\n1. Filter iris flowers with sepal length > 6.0 and petal width > 1.5")
challenge1 = iris_df[(iris_df['sepal length (cm)'] > 6.0) & 
                     (iris_df['petal width (cm)'] > 1.5)]
print(f"Found {len(challenge1)} flowers")
print(challenge1[['sepal length (cm)', 'petal width (cm)', 'species_name']])

print("\n2. Create a new column: sepal_area = sepal_length * sepal_width")
iris_df['sepal_area'] = iris_df['sepal length (cm)'] * iris_df['sepal width (cm)']
print("Top 5 by sepal area:")
print(iris_df.nlargest(5, 'sepal_area')[['sepal length (cm)', 
                                          'sepal width (cm)', 
                                          'sepal_area', 
                                          'species_name']])

print("\n3. Find correlation between features")
correlations = iris_df[['sepal length (cm)', 'sepal width (cm)', 
                        'petal length (cm)', 'petal width (cm)']].corr()
print(correlations)

print("\n" + "="*60)
print("Pandas exercises complete!")
print("Next: Complete the Week 1 project")
print("="*60)