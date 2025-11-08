"""
WEEK 1 FINAL PROJECT: Student Performance Analyzer
==================================================
Build a complete data analysis pipeline from scratch

Save as: D:\ai_engineering\week1_data_handling\final_project.py

OBJECTIVE:
Analyze student exam scores, identify patterns, and create insights.
This combines NumPy arrays, Pandas DataFrames, and visualizations.

TASKS:
1. Generate synthetic student data
2. Clean and preprocess data
3. Perform statistical analysis
4. Create visualizations
5. Generate a summary report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("="*70)
print("WEEK 1 FINAL PROJECT: Student Performance Analyzer")
print("="*70)

# ============================================
# TASK 1: Generate Synthetic Student Data
# ============================================
print("\n>>> TASK 1: Generating Student Data")

np.random.seed(42)  # For reproducibility

# Number of students
n_students = 100

# Generate student data
students = {
    'student_id': [f'S{i:04d}' for i in range(1, n_students + 1)],
    'name': [f'Student_{i}' for i in range(1, n_students + 1)],
    'age': np.random.randint(18, 25, n_students),
    'gender': np.random.choice(['M', 'F'], n_students),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_students),
    'math_score': np.random.randint(40, 100, n_students),
    'science_score': np.random.randint(35, 100, n_students),
    'english_score': np.random.randint(45, 100, n_students),
    'study_hours': np.random.randint(1, 10, n_students),
    'attendance': np.random.randint(60, 100, n_students)
}

# Create DataFrame
df = pd.DataFrame(students)

# Introduce some missing values (realistic scenario)
missing_indices = np.random.choice(n_students, 5, replace=False)
df.loc[missing_indices[:3], 'math_score'] = np.nan
df.loc[missing_indices[3:], 'attendance'] = np.nan

print(f"✅ Generated data for {n_students} students")
print("\nFirst 5 rows:")
print(df.head())

# Save raw data
df.to_csv('D:/ai_engineering/datasets/students_raw.csv', index=False)
print("\n✅ Raw data saved: students_raw.csv")

# ============================================
# TASK 2: Data Cleaning and Preprocessing
# ============================================
print("\n>>> TASK 2: Data Cleaning")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values with median
df['math_score'].fillna(df['math_score'].median(), inplace=True)
df['attendance'].fillna(df['attendance'].median(), inplace=True)

print("\n✅ Missing values filled with median")

# Add calculated columns
df['total_score'] = df['math_score'] + df['science_score'] + df['english_score']
df['average_score'] = df['total_score'] / 3
df['performance_category'] = pd.cut(df['average_score'], 
                                     bins=[0, 50, 70, 85, 100],
                                     labels=['Poor', 'Average', 'Good', 'Excellent'])

print("\n✅ Added calculated columns: total_score, average_score, performance_category")
print("\nUpdated DataFrame:")
print(df.head())

# Save cleaned data
df.to_csv('D:/ai_engineering/datasets/students_cleaned.csv', index=False)
print("\n✅ Cleaned data saved: students_cleaned.csv")

# ============================================
# TASK 3: Statistical Analysis
# ============================================
print("\n>>> TASK 3: Statistical Analysis")

print("\n--- Basic Statistics ---")
print(df[['math_score', 'science_score', 'english_score', 'average_score']].describe())

print("\n--- Performance by Gender ---")
gender_stats = df.groupby('gender').agg({
    'math_score': 'mean',
    'science_score': 'mean',
    'english_score': 'mean',
    'average_score': 'mean',
    'study_hours': 'mean'
}).round(2)
print(gender_stats)

print("\n--- Performance by City ---")
city_stats = df.groupby('city')['average_score'].agg(['mean', 'min', 'max', 'count']).round(2)
print(city_stats)

print("\n--- Performance Categories Distribution ---")
print(df['performance_category'].value_counts())

print("\n--- Top 10 Students ---")
top_students = df.nlargest(10, 'average_score')[['name', 'average_score', 
                                                   'study_hours', 'attendance']]
print(top_students)

print("\n--- Correlation Analysis ---")
correlation = df[['math_score', 'science_score', 'english_score', 
                   'study_hours', 'attendance']].corr()
print(correlation.round(3))

# ============================================
# TASK 4: Data Insights with NumPy
# ============================================
print("\n>>> TASK 4: Advanced Analysis with NumPy")

# Convert scores to NumPy arrays for advanced operations
math_scores = df['math_score'].values
science_scores = df['science_score'].values
english_scores = df['english_score'].values

# Calculate percentiles
print("\n--- Score Percentiles ---")
for subject, scores in [('Math', math_scores), 
                         ('Science', science_scores), 
                         ('English', english_scores)]:
    p25 = np.percentile(scores, 25)
    p50 = np.percentile(scores, 50)
    p75 = np.percentile(scores, 75)
    print(f"{subject}: 25th={p25:.1f}, 50th={p50:.1f}, 75th={p75:.1f}")

# Find students who excel in all subjects (> 80 in all)
high_performers = df[(df['math_score'] > 80) & 
                     (df['science_score'] > 80) & 
                     (df['english_score'] > 80)]
print(f"\n✨ High performers (>80 in all subjects): {len(high_performers)}")

# Find students who need help (< 50 in any subject)
struggling = df[(df['math_score'] < 50) | 
                (df['science_score'] < 50) | 
                (df['english_score'] < 50)]
print(f"⚠️  Students needing help (<50 in any subject): {len(struggling)}")

# Analyze relationship between study hours and performance
study_groups = df.groupby(pd.cut(df['study_hours'], bins=[0, 3, 6, 10]))['average_score'].mean()
print("\n--- Average Score by Study Hours ---")
print(study_groups.round(2))

# ============================================
# TASK 5: Visualizations
# ============================================
print("\n>>> TASK 5: Creating Visualizations")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Score Distribution (Histogram)
ax1 = plt.subplot(3, 3, 1)
ax1.hist(df['math_score'], bins=15, alpha=0.7, color='blue', edgecolor='black')
ax1.set_xlabel('Math Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Math Score Distribution')
ax1.axvline(df['math_score'].mean(), color='red', linestyle='--', label='Mean')
ax1.legend()

ax2 = plt.subplot(3, 3, 2)
ax2.hist(df['science_score'], bins=15, alpha=0.7, color='green', edgecolor='black')
ax2.set_xlabel('Science Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Science Score Distribution')
ax2.axvline(df['science_score'].mean(), color='red', linestyle='--', label='Mean')
ax2.legend()

ax3 = plt.subplot(3, 3, 3)
ax3.hist(df['english_score'], bins=15, alpha=0.7, color='orange', edgecolor='black')
ax3.set_xlabel('English Score')
ax3.set_ylabel('Frequency')
ax3.set_title('English Score Distribution')
ax3.axvline(df['english_score'].mean(), color='red', linestyle='--', label='Mean')
ax3.legend()

# Plot 4: Performance Category Pie Chart
ax4 = plt.subplot(3, 3, 4)
category_counts = df['performance_category'].value_counts()
ax4.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
ax4.set_title('Performance Categories')

# Plot 5: Average Score by City (Bar Chart)
ax5 = plt.subplot(3, 3, 5)
city_avg = df.groupby('city')['average_score'].mean().sort_values(ascending=False)
ax5.bar(city_avg.index, city_avg.values, color=['red', 'blue', 'green', 'orange', 'purple'])
ax5.set_xlabel('City')
ax5.set_ylabel('Average Score')
ax5.set_title('Average Score by City')
ax5.tick_params(axis='x', rotation=45)

# Plot 6: Study Hours vs Average Score (Scatter)
ax6 = plt.subplot(3, 3, 6)
scatter = ax6.scatter(df['study_hours'], df['average_score'], 
                      c=df['attendance'], cmap='viridis', alpha=0.6)
ax6.set_xlabel('Study Hours')
ax6.set_ylabel('Average Score')
ax6.set_title('Study Hours vs Performance')
plt.colorbar(scatter, ax=ax6, label='Attendance %')

# Plot 7: Attendance vs Average Score
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(df['attendance'], df['average_score'], alpha=0.5, color='purple')
ax7.set_xlabel('Attendance %')
ax7.set_ylabel('Average Score')
ax7.set_title('Attendance vs Performance')

# Plot 8: Box Plot - Scores by Gender
ax8 = plt.subplot(3, 3, 8)
df.boxplot(column='average_score', by='gender', ax=ax8)
ax8.set_xlabel('Gender')
ax8.set_ylabel('Average Score')
ax8.set_title('Score Distribution by Gender')
plt.suptitle('')  # Remove default title

# Plot 9: Correlation Heatmap
ax9 = plt.subplot(3, 3, 9)
corr_data = df[['math_score', 'science_score', 'english_score', 
                'study_hours', 'attendance']].corr()
im = ax9.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax9.set_xticks(range(len(corr_data.columns)))
ax9.set_yticks(range(len(corr_data.columns)))
ax9.set_xticklabels(['Math', 'Science', 'English', 'Study', 'Attend'], rotation=45)
ax9.set_yticklabels(['Math', 'Science', 'English', 'Study', 'Attend'])
ax9.set_title('Correlation Heatmap')
plt.colorbar(im, ax=ax9)

# Add correlation values
for i in range(len(corr_data)):
    for j in range(len(corr_data)):
        text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week1_data_handling/student_analysis.png', dpi=150)
print("\n✅ Visualization saved: student_analysis.png")
plt.close()

# ============================================
# TASK 6: Generate Summary Report
# ============================================
print("\n>>> TASK 6: Generating Summary Report")

report = f"""
{'='*70}
                    STUDENT PERFORMANCE REPORT
{'='*70}

DATASET OVERVIEW:
- Total Students: {len(df)}
- Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATISTICS:
- Average Math Score: {df['math_score'].mean():.2f}
- Average Science Score: {df['science_score'].mean():.2f}
- Average English Score: {df['english_score'].mean():.2f}
- Overall Average: {df['average_score'].mean():.2f}

PERFORMANCE BREAKDOWN:
{df['performance_category'].value_counts().to_string()}

TOP 5 STUDENTS:
{df.nlargest(5, 'average_score')[['name', 'average_score', 'study_hours']].to_string(index=False)}

KEY INSIGHTS:
1. Correlation between study hours and performance: {df[['study_hours', 'average_score']].corr().iloc[0,1]:.3f}
2. Correlation between attendance and performance: {df[['attendance', 'average_score']].corr().iloc[0,1]:.3f}
3. Best performing city: {df.groupby('city')['average_score'].mean().idxmax()}
4. Students excelling in all subjects: {len(high_performers)}
5. Students needing additional support: {len(struggling)}

RECOMMENDATIONS:
- Focus on students with <50% attendance
- Provide additional tutoring for students scoring <50
- Encourage study hours (strong positive correlation observed)
- Investigate teaching methods in top-performing cities

{'='*70}
"""

print(report)

# Save report
with open('D:/ai_engineering/week1_data_handling/analysis_report.txt', 'w') as f:
    f.write(report)

print("\n✅ Report saved: analysis_report.txt")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("PROJECT COMPLETE! 🎉")
print("="*70)
print("\nFiles Generated:")
print("1. students_raw.csv - Original generated data")
print("2. students_cleaned.csv - Processed data")
print("3. student_analysis.png - 9 visualizations")
print("4. analysis_report.txt - Summary report")
print("\n" + "="*70)
print("SKILLS DEMONSTRATED:")
print("✅ NumPy array operations and statistics")
print("✅ Pandas data manipulation and grouping")
print("✅ Data cleaning (handling missing values)")
print("✅ Statistical analysis (correlation, percentiles)")
print("✅ Data visualization (9 different plot types)")
print("✅ Report generation")
print("="*70)