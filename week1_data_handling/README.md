# 📊 Week 1: Data Handling Fundamentals

**Duration:** ~10-12 hours  
**Status:** ✅ Complete  
**Goal:** Master NumPy, Pandas, and data visualization

---

## 📚 Learning Objectives

By the end of Week 1, you should be able to:
- ✅ Manipulate arrays with NumPy
- ✅ Clean and analyze data with Pandas
- ✅ Create meaningful visualizations
- ✅ Generate statistical reports
- ✅ Handle missing data
- ✅ Perform exploratory data analysis (EDA)

---

## 📁 Files in This Directory

```
week1_data_handling/
├── README.md                    # This file
├── numpy_practice.py            # NumPy fundamentals
├── pandas_practice.py           # Pandas operations
├── final_project.py             # Student Performance Analyzer
├── numpy_image.png              # Generated visualization
├── iris_analysis.png            # Iris dataset analysis
├── student_analysis.png         # Final project output
└── analysis_report.txt          # Text summary report
```

---

## 🎯 Day-by-Day Breakdown

### **Day 1-2: NumPy Fundamentals**
**File:** `numpy_practice.py`

**Topics Covered:**
- Array creation and indexing
- Element-wise operations
- Aggregation functions (sum, mean, std)
- Reshaping and slicing
- Boolean indexing
- Image representation as arrays

**Key Exercises:**
```python
# Array creation
arr = np.arange(0, 10, 2)
matrix = np.zeros((3, 3))

# Operations
result = arr ** 2
mean_val = np.mean(arr)

# Indexing
subset = arr[arr > 4]
```

**Output:**
- Created 100x100 image with NumPy
- Practiced statistical operations
- Mastered array manipulation

---

### **Day 3-4: Pandas for Data Analysis**
**File:** `pandas_practice.py`

**Topics Covered:**
- DataFrame creation and exploration
- Data selection and filtering
- Grouping and aggregation
- Handling missing values
- Merging and joining data
- Data visualization

**Key Operations:**
```python
# Load data
df = pd.read_csv('data.csv')

# Filter data
high_scorers = df[df['score'] > 80]

# Group and aggregate
avg_by_category = df.groupby('category')['score'].mean()

# Handle missing values
df.fillna(df.mean(), inplace=True)
```

**Dataset Used:**
- **Iris Dataset** (150 samples, 4 features)
- Classic ML dataset for flowers classification
- Built-in to scikit-learn

**Output:**
- 4-panel visualization: histograms, scatter plots, box plots, bar charts
- Statistical summary of all features
- Correlation analysis

---

### **Day 5-7: Final Project - Student Performance Analyzer**
**File:** `final_project.py`

**Project Description:**
Comprehensive data analysis pipeline that generates synthetic student data, cleans it, performs statistical analysis, and creates visualizations.

**Dataset:**
- **Size**: 100 students
- **Features**: 
  - Demographics: Age, Gender, City
  - Scores: Math, Science, English
  - Behavior: Study hours, Attendance
- **Generated synthetically** with realistic distributions

**Pipeline Steps:**

1. **Data Generation**
   - Create 100 student records
   - Add missing values (realistic scenario)
   
2. **Data Cleaning**
   - Fill missing values with median
   - Create calculated columns (total score, average)
   - Categorize performance (Poor/Average/Good/Excellent)

3. **Statistical Analysis**
   - Descriptive statistics
   - Performance by gender
   - Performance by city
   - Correlation analysis
   - Top performers identification

4. **Visualizations** (9 panels)
   - Score distributions (histograms)
   - Performance categories (pie chart)
   - City comparison (bar chart)
   - Study hours vs performance (scatter)
   - Attendance impact (scatter)
   - Gender comparison (box plot)
   - Correlation heatmap

5. **Report Generation**
   - Summary statistics
   - Key insights
   - Recommendations

**Key Code Snippets:**
```python
# Generate data with missing values
students = {
    'student_id': [f'S{i:04d}' for i in range(1, 101)],
    'math_score': np.random.randint(40, 100, 100),
    # ... more features
}
df = pd.DataFrame(students)

# Clean data
df['math_score'].fillna(df['math_score'].median(), inplace=True)

# Analyze
correlation = df[['study_hours', 'average_score']].corr()
top_students = df.nlargest(10, 'average_score')

# Visualize
plt.scatter(df['study_hours'], df['average_score'])
plt.xlabel('Study Hours')
plt.ylabel('Average Score')
```

**Outputs:**
- `students_raw.csv` - Original generated data
- `students_cleaned.csv` - Processed data
- `student_analysis.png` - 9-panel visualization
- `analysis_report.txt` - Text summary

**Key Findings:**
- Strong positive correlation between study hours and performance (0.65+)
- Attendance significantly impacts scores
- City-based performance variations
- 15-20% students need additional support

---

## 📊 Skills Demonstrated

### **NumPy Skills**
- ✅ Array creation and manipulation
- ✅ Mathematical operations
- ✅ Statistical computations
- ✅ Boolean indexing
- ✅ Reshaping and broadcasting

### **Pandas Skills**
- ✅ DataFrame operations
- ✅ Data cleaning
- ✅ Grouping and aggregation
- ✅ Missing value handling
- ✅ Data visualization
- ✅ CSV I/O operations

### **Visualization Skills**
- ✅ Histograms
- ✅ Scatter plots
- ✅ Box plots
- ✅ Bar charts
- ✅ Pie charts
- ✅ Heatmaps
- ✅ Multi-panel layouts

### **Analysis Skills**
- ✅ Descriptive statistics
- ✅ Correlation analysis
- ✅ Data categorization
- ✅ Insight generation
- ✅ Report writing

---

## 🚀 How to Run

### **Prerequisites**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Execute Scripts**
```bash
# Activate virtual environment
cd D:\ai_engineering
ai_env\Scripts\activate

# Run NumPy practice
python week1_data_handling/numpy_practice.py

# Run Pandas practice
python week1_data_handling/pandas_practice.py

# Run final project
python week1_data_handling/final_project.py
```

### **Expected Runtime**
- NumPy practice: ~30 seconds
- Pandas practice: ~1 minute
- Final project: ~2-3 minutes

---

## 📈 Results & Metrics

### **Generated Files**
| File | Size | Description |
|------|------|-------------|
| `students_raw.csv` | ~8 KB | Original synthetic data |
| `students_cleaned.csv` | ~9 KB | Processed data |
| `iris_processed.csv` | ~5 KB | Iris analysis output |
| `student_analysis.png` | ~450 KB | 9-panel visualization |
| `iris_analysis.png` | ~400 KB | 4-panel Iris plots |
| `analysis_report.txt` | ~2 KB | Text summary |

### **Performance**
- Data processing: < 1 second
- Visualization generation: ~2 seconds
- Total execution time: ~3 minutes

---

## 🎓 Key Takeaways

### **Technical Learnings**
1. **NumPy** is fundamental for numerical computing
2. **Pandas** makes data manipulation intuitive
3. **Visualization** reveals patterns hidden in numbers
4. **Data cleaning** is 80% of the work
5. **Statistical analysis** drives decision-making

### **Best Practices Learned**
- Always explore data before analysis
- Handle missing values appropriately
- Visualize data from multiple angles
- Document insights clearly
- Save intermediate results

### **Common Pitfalls Avoided**
- Not checking for missing values
- Ignoring data types
- Over-complicating visualizations
- Forgetting to save outputs
- Not validating results

---

## 🔄 Exercises to Extend Learning

Want to practice more? Try these challenges:

1. **Modify the dataset**
   - Change student count to 500
   - Add more subjects (History, Geography)
   - Include GPA calculation

2. **Advanced analysis**
   - Predict final grades based on attendance
   - Identify students at risk of failing
   - Create student profiles

3. **Enhanced visualizations**
   - Interactive plots with Plotly
   - Dashboard with multiple pages
   - Animated time-series plots

4. **Real datasets**
   - Download Titanic dataset from Kaggle
   - Analyze COVID-19 data
   - Explore census data

---

## 🔗 Resources

### **Documentation**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### **Datasets for Practice**
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Data.gov](https://data.gov/)

### **Cheat Sheets**
- [NumPy Cheat Sheet](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

---

## ✅ Completion Checklist

- [x] Completed `numpy_practice.py`
- [x] Completed `pandas_practice.py`
- [x] Completed `final_project.py`
- [x] Generated all visualizations
- [x] Understood all statistical measures
- [x] Reviewed and interpreted results
- [ ] Extended project with custom data
- [ ] Tried real-world datasets

---

## ➡️ Next Steps

**Ready for Week 2?**
- Head to [week2_ml_basics/README.md](../week2_ml_basics/README.md)
- Learn Machine Learning algorithms
- Build classification and regression models
- Create your first spam detector!

---

**Week 1 Status:** ✅ Complete  
**Time Invested:** ~10-12 hours  
**Scripts Mastered:** 3  
**Visualizations Created:** 15+  
**Ready for ML:** Yes! 🚀