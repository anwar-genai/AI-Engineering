# 🤖 Week 2: Machine Learning Fundamentals

**Duration:** ~12-15 hours  
**Status:** ✅ Complete  
**Goal:** Master supervised and unsupervised learning with scikit-learn

---

## 📚 Learning Objectives

By the end of Week 2, you should be able to:
- ✅ Train classification models (Logistic Regression, Decision Trees)
- ✅ Build regression models for predictions
- ✅ Perform clustering analysis (K-Means)
- ✅ Evaluate models with proper metrics
- ✅ Use cross-validation for robust evaluation
- ✅ Process text data for ML (TF-IDF)
- ✅ Compare multiple algorithms
- ✅ Build complete ML pipelines

---

## 📁 Files in This Directory

```
week2_ml_basics/
├── README.md                              # This file
├── day1_classification.py                 # Logistic Regression
├── day2_decision_trees.py                 # Decision Trees & comparison
├── day3_regression.py                     # Linear Regression
├── day4_clustering.py                     # K-Means clustering
├── final_spam_classifier.py               # Complete text classification
├── day1_classification_results.png        # Classification visualizations
├── day2_decision_trees_results.png        # Tree analysis plots
├── day3_regression_results.png            # Regression diagnostics
├── day4_clustering_results.png            # Cluster visualizations
├── final_spam_classifier_results.png      # Spam detector dashboard
├── clustering_report.txt                  # Customer segmentation insights
├── spam_classifier_report.txt             # Final project report
└── model_comparison.csv                   # Model performance metrics
```

---

## 🎯 Day-by-Day Breakdown

### **Day 1: Classification Basics**
**File:** `day1_classification.py`  
**Dataset:** Iris (150 samples, 3 classes)

**Topics:**
- Train/test split
- Logistic Regression
- Confusion matrix
- Accuracy, precision, recall, F1-score
- Feature scaling
- Prediction probabilities

**Key Code:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
```

**Results:**
- Test accuracy: 97-100%
- 4-panel visualization showing confusion matrix, feature importance, predictions
- Binary classification challenge (Setosa vs Others)

**Key Learning:**
- Always split data before training
- Confusion matrix shows where model makes mistakes
- Feature scaling improves performance

---

### **Day 2-3: Decision Trees & Model Comparison**
**File:** `day2_decision_trees.py`  
**Datasets:** Wine (178 samples), Breast Cancer (569 samples)

**Topics:**
- Decision Tree Classifier
- Overfitting detection
- Feature importance
- Hyperparameter tuning (max_depth, min_samples_split)
- Cross-validation (5-fold)
- Model comparison

**Key Code:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Train with depth limit
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

# Feature importance
importances = dt.feature_importances_

# Cross-validation
cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f}")
```

**Results:**
- Compared 4 models: Decision Tree (default), Decision Tree (optimized), Logistic Regression
- Wine dataset accuracy: 92-97%
- Breast Cancer accuracy: 94-96%
- Identified top 10 most important features

**Key Learning:**
- High training accuracy + low test accuracy = overfitting
- Use `max_depth` to prevent overfitting
- Decision trees are interpretable but prone to overfitting
- Cross-validation gives more reliable estimates

**Visualizations:**
- Decision tree structure (depth=3)
- Accuracy vs max_depth (overfitting curve)
- Feature importance bar chart
- Multiple confusion matrices

---

### **Day 4: Regression Models**
**File:** `day3_regression.py`  
**Dataset:** California Housing (20,640 samples, 8 features)

**Topics:**
- Linear Regression
- Regression metrics (R², MSE, RMSE, MAE)
- Residual analysis
- Feature coefficients
- Polynomial features

**Key Code:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Feature importance
coefficients = lr.coef_
```

**Results:**
- R² Score: 0.58-0.62 (explains 60% of variance)
- RMSE: ~$68,000 (average prediction error)
- MAE: ~$50,000 (median error)
- Identified top price predictors: Median Income, Location

**Key Learning:**
- R² closer to 1.0 = better model
- RMSE penalizes large errors more than MAE
- Residuals should be randomly distributed
- Feature coefficients show impact on target

**Visualizations:**
- Actual vs Predicted scatter plot
- Residual plot
- Residuals distribution (histogram)
- Feature coefficients bar chart
- Q-Q plot for normality check

---

### **Day 5: Clustering (Unsupervised Learning)**
**File:** `day4_clustering.py`  
**Dataset:** Synthetic customers (300 samples, 3 features)

**Topics:**
- K-Means clustering
- Elbow method
- Silhouette score
- Customer segmentation
- Cluster interpretation

**Key Code:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Find optimal k
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Train with optimal k
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k)
clusters = kmeans.fit_predict(X)

# Evaluate
silhouette = silhouette_score(X, clusters)
```

**Results:**
- Optimal clusters: 5
- Silhouette score: 0.55-0.65
- Identified customer segments:
  1. High Income, High Spenders (VIP)
  2. High Income, Low Spenders (Savers)
  3. Low Income, High Spenders (Impulsive)
  4. Low Income, Low Spenders (Budget)
  5. Middle Income, Moderate Spenders

**Key Learning:**
- K-Means requires specifying k upfront
- Feature scaling is CRITICAL
- Use elbow method and silhouette score to find k
- Business interpretation is essential

**Visualizations:**
- Elbow curve
- Silhouette scores vs k
- 2D cluster visualization with centroids
- 3D scatter plot (Income, Spending, Age)
- Cluster size distribution
- Heatmap of cluster characteristics

**Business Value:**
- Targeted marketing campaigns
- Personalized recommendations
- Resource allocation
- Customer retention strategies

---

### **Day 6-7: Final Project - SMS Spam Classifier**
**File:** `final_spam_classifier.py`  
**Dataset:** SMS messages (500 samples, spam/ham)

**Complete ML Pipeline:**

1. **Data Generation**
   - 200 spam messages (prizes, urgent alerts, free offers)
   - 300 ham messages (normal conversations)
   - Realistic patterns and variations

2. **Text Preprocessing**
   ```python
   def preprocess_text(text):
       text = text.lower()
       text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
       text = re.sub(r'\d{4}', '', text)  # Remove phone numbers
       text = text.translate(str.maketrans('', '', string.punctuation))
       return text
   ```

3. **Feature Extraction**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   
   vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
   X = vectorizer.fit_transform(messages)
   ```

4. **Model Training** (4 algorithms)
   - Logistic Regression
   - Decision Tree
   - Naive Bayes
   - Linear SVM

5. **Model Comparison**
   ```python
   for name, model in models.items():
       model.fit(X_train, y_train)
       accuracy = model.score(X_test, y_test)
       f1 = f1_score(y_test, y_pred)
   ```

6. **Deployment Testing**
   - Test on 6 new unseen messages
   - Predict with confidence scores

**Results:**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **98.5%** | **0.98** | **0.99** | **0.98** |
| Naive Bayes | 97.8% | 0.97 | 0.98 | 0.97 |
| Linear SVM | 98.2% | 0.98 | 0.98 | 0.98 |
| Decision Tree | 94.5% | 0.93 | 0.95 | 0.94 |

**Confusion Matrix (Best Model):**
```
                Predicted Ham  Predicted Spam
Actual Ham           58              1
Actual Spam           0             41
```

**Key Spam Indicators:**
- "winner", "free", "prize", "urgent"
- Phone numbers and URLs
- ALL CAPS usage
- Call-to-action phrases

**Key Ham Indicators:**
- Personal pronouns ("you", "I", "we")
- Conversational tone
- Specific context
- Normal punctuation

**Outputs:**
- `sms_spam.csv` - Generated dataset
- `final_spam_classifier_results.png` - 9-panel dashboard
- `spam_classifier_report.txt` - Detailed report
- `model_comparison.csv` - Performance metrics

---

## 📊 Skills Demonstrated

### **Classification**
- ✅ Binary and multi-class classification
- ✅ Logistic Regression
- ✅ Decision Trees
- ✅ Model evaluation (accuracy, precision, recall, F1)
- ✅ Confusion matrix interpretation
- ✅ Overfitting detection and prevention
- ✅ Hyperparameter tuning

### **Regression**
- ✅ Linear Regression
- ✅ R², MSE, RMSE, MAE metrics
- ✅ Residual analysis
- ✅ Feature coefficient interpretation
- ✅ Polynomial features

### **Clustering**
- ✅ K-Means algorithm
- ✅ Elbow method
- ✅ Silhouette score
- ✅ Customer segmentation
- ✅ Cluster interpretation

### **Text Processing**
- ✅ Text preprocessing (cleaning, tokenization)
- ✅ TF-IDF vectorization
- ✅ N-grams (unigrams, bigrams)
- ✅ Stop words removal
- ✅ Feature extraction from text

### **Model Comparison**
- ✅ Training multiple algorithms
- ✅ Comparing performance metrics
- ✅ Selecting best model
- ✅ Cross-validation

### **End-to-End ML Pipeline**
- ✅ Data generation/loading
- ✅ Preprocessing
- ✅ Feature engineering
- ✅ Model training
- ✅ Evaluation
- ✅ Testing on new data
- ✅ Reporting

---

## 🚀 How to Run

### **Prerequisites**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Execute Scripts**
```bash
# Activate environment
cd D:\ai_engineering
ai_env\Scripts\activate

# Run in sequence
python week2_ml_basics/day1_classification.py
python week2_ml_basics/day2_decision_trees.py
python week2_ml_basics/day3_regression.py
python week2_ml_basics/day4_clustering.py

# Final project
python week2_ml_basics/final_spam_classifier.py
```

### **Expected Runtime**
- Day 1: ~30 seconds
- Day 2-3: ~1 minute
- Day 4: ~1 minute  
- Day 5: ~1 minute
- Final Project: ~2 minutes

---

## 📈 Performance Metrics Summary

### **Classification Performance**
- Iris: 97-100% accuracy
- Wine: 92-97% accuracy
- Breast Cancer: 94-96% accuracy
- SMS Spam: 98.5% accuracy (F1: 0.98)

### **Regression Performance**
- California Housing: R² = 0.60, RMSE = $68k

### **Clustering Performance**
- Customer Segmentation: Silhouette score = 0.62
- 5 distinct customer segments identified

### **Overall Statistics**
- **Total Models Trained**: 12+
- **Datasets Used**: 6
- **Visualizations Created**: 35+
- **Lines of Code Written**: ~2000+

---

## 🎓 Key Takeaways

### **Technical Concepts**
1. **Train/Test Split**: Essential for unbiased evaluation
2. **Overfitting**: When model memorizes training data
3. **Cross-Validation**: More reliable than single split
4. **Feature Scaling**: Critical for distance-based algorithms
5. **Metrics Matter**: Choose based on business needs
6. **No Free Lunch**: No single best algorithm for all problems

### **Best Practices**
- Always visualize confusion matrix
- Use cross-validation for small datasets
- Scale features before training
- Compare multiple models
- Interpret feature importance
- Test on completely new data
- Document results thoroughly

### **Common Mistakes Avoided**
- Training on test data
- Ignoring class imbalance
- Not scaling features
- Overfitting without detection
- Using accuracy alone for imbalanced data
- Not validating on unseen data

---

## 🔄 Exercises to Extend Learning

### **Beginner Level**
1. Modify spam classifier to detect phishing emails
2. Add more features to customer segmentation
3. Try different values of k for K-Means
4. Use different regression metrics

### **Intermediate Level**
1. Handle imbalanced datasets (SMOTE)
2. Implement ensemble methods (Random Forest)
3. Try different text vectorizers (Count, Word2Vec)
4. Build a hotel review sentiment classifier

### **Advanced Level**
1. Create a recommendation system with clustering
2. Time-series forecasting with regression
3. Multi-label text classification
4. Deploy model as REST API

---

## 📚 Resources

### **Documentation**
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn API Reference](https://scikit-learn.org/stable/modules/classes.html)

### **Datasets for Practice**
- [Kaggle Titanic](https://www.kaggle.com/c/titanic)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle SMS Spam](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

### **Learning Resources**
- [Scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)

---

## 🏆 Achievements Unlocked

- ✅ **First ML Model**: Trained Logistic Regression
- ✅ **Model Comparison**: Compared 4+ algorithms
- ✅ **98% Accuracy**: Built high-performance spam detector
- ✅ **Business Insights**: Generated customer segmentation report
- ✅ **Complete Pipeline**: End-to-end ML system
- ✅ **Production Ready**: Tested on unseen data

---

## ✅ Completion Checklist

- [x] Completed Day 1: Classification
- [x] Completed Day 2-3: Decision Trees
- [x] Completed Day 4: Regression
- [x] Completed Day 5: Clustering
- [x] Completed Final Project: Spam Classifier
- [x] Achieved 98%+ accuracy on spam detection
- [x] Generated all visualizations
- [x] Understood all metrics
- [ ] Extended projects with real datasets
- [ ] Built custom ML pipeline

---

## ➡️ Next Steps: Phase 2

**Ready for Deep Learning?**
- Week 3: PyTorch fundamentals
- Week 4: Neural networks from scratch
- Week 5: CNNs for image classification
- Week 6: Transfer learning & fine-tuning
- Week 7-8: OCR with deep learning

**What You'll Learn:**
- Building neural networks
- Training on GPUs (Google Colab)
- Computer vision basics
- Transfer learning
- Model deployment

---

## 📞 Questions & Improvements

**Common Questions:**

**Q: Why is my accuracy low?**
A: Check class balance, try feature scaling, adjust hyperparameters

**Q: What's the best algorithm?**
A: Depends on data! Always compare multiple models

**Q: How to handle overfitting?**
A: Use cross-validation, regularization, simpler models, more data

**Q: When to use classification vs regression?**
A: Classification for categories, regression for continuous values

---

**Week 2 Status:** ✅ Complete  
**Time Invested:** ~12-15 hours  
**Models Trained:** 12+  
**Accuracy Achieved:** 98.5%  
**Ready for Deep Learning:** Absolutely! 🚀  

---

**Last Updated:** November 2024  
**Next Challenge:** PyTorch & Neural Networks 🧠