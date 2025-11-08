"""
WEEK 2 - DAY 2-3: Decision Trees & Model Comparison
===================================================
Learn Decision Trees and compare with Logistic Regression



Topics:
- Decision Tree Classifier
- Feature importance
- Overfitting vs Underfitting
- Cross-validation
- Model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

print("="*70)
print("WEEK 2 - DAY 2-3: Decision Trees")
print("="*70)

# ============================================
# STEP 1: Load Wine Quality Dataset
# ============================================
print("\n>>> STEP 1: Loading Wine Dataset")

wine = load_wine()
X = wine.data
y = wine.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Feature names: {wine.feature_names}")
print(f"Target names: {wine.target_names}")

# Create DataFrame
df = pd.DataFrame(X, columns=wine.feature_names)
df['wine_class'] = y

print("\nFirst 5 samples:")
print(df.head())

print("\nClass distribution:")
print(pd.Series(y).value_counts())

# ============================================
# STEP 2: Train-Test Split
# ============================================
print("\n>>> STEP 2: Splitting Data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================
# STEP 3: Train Decision Tree (Default)
# ============================================
print("\n>>> STEP 3: Training Decision Tree (Default Parameters)")

# Create and train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Accuracy
train_acc_dt = dt_model.score(X_train, y_train)
test_acc_dt = dt_model.score(X_test, y_test)

print(f"Training Accuracy: {train_acc_dt:.4f}")
print(f"Test Accuracy: {test_acc_dt:.4f}")

# Note: If training accuracy is much higher than test, it's overfitting!
if train_acc_dt - test_acc_dt > 0.1:
    print("⚠️  Warning: Possible overfitting detected!")

# ============================================
# STEP 4: Feature Importance
# ============================================
print("\n>>> STEP 4: Analyzing Feature Importance")

# Get feature importances
importances = dt_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

# ============================================
# STEP 5: Control Overfitting with max_depth
# ============================================
print("\n>>> STEP 5: Preventing Overfitting")

# Try different max_depth values
depths = [2, 3, 5, 10, 15, None]
results = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    
    results.append({
        'max_depth': depth,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'difference': train_acc - test_acc
    })
    
results_df = pd.DataFrame(results)
print("\nAccuracy vs max_depth:")
print(results_df)

# Find best depth (highest test accuracy)
best_depth = results_df.loc[results_df['test_acc'].idxmax(), 'max_depth']
print(f"\n✅ Best max_depth: {best_depth}")

# ============================================
# STEP 6: Train Optimized Decision Tree
# ============================================
print("\n>>> STEP 6: Training Optimized Decision Tree")

dt_optimized = DecisionTreeClassifier(
    max_depth=5,  # Prevent overfitting
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=5,  # Minimum samples in leaf
    random_state=42
)
dt_optimized.fit(X_train, y_train)

train_acc_opt = dt_optimized.score(X_train, y_train)
test_acc_opt = dt_optimized.score(X_test, y_test)

print(f"Optimized Training Accuracy: {train_acc_opt:.4f}")
print(f"Optimized Test Accuracy: {test_acc_opt:.4f}")
print(f"Difference: {train_acc_opt - test_acc_opt:.4f}")

# ============================================
# STEP 7: Cross-Validation
# ============================================
print("\n>>> STEP 7: Cross-Validation (5-Fold)")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(dt_optimized, X_train, y_train, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================
# STEP 8: Compare with Logistic Regression
# ============================================
print("\n>>> STEP 8: Model Comparison")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Compare accuracies
comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Decision Tree (Optimized)', 'Logistic Regression'],
    'Train Accuracy': [train_acc_dt, train_acc_opt, lr_model.score(X_train_scaled, y_train)],
    'Test Accuracy': [test_acc_dt, test_acc_opt, lr_model.score(X_test_scaled, y_test)]
})

print("\n" + "="*60)
print(comparison)
print("="*60)

# ============================================
# STEP 9: Visualizations
# ============================================
print("\n>>> STEP 9: Creating Visualizations")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Feature Importance
ax1 = plt.subplot(3, 3, 1)
top_features = feature_importance_df.head(10)
ax1.barh(top_features['feature'], top_features['importance'], color='steelblue')
ax1.set_xlabel('Importance')
ax1.set_title('Top 10 Feature Importances')
ax1.invert_yaxis()

# Plot 2: Accuracy vs max_depth
ax2 = plt.subplot(3, 3, 2)
depths_plot = [str(d) if d is not None else 'None' for d in results_df['max_depth']]
ax2.plot(depths_plot, results_df['train_acc'], marker='o', label='Train', linewidth=2)
ax2.plot(depths_plot, results_df['test_acc'], marker='s', label='Test', linewidth=2)
ax2.set_xlabel('max_depth')
ax2.set_ylabel('Accuracy')
ax2.set_title('Overfitting: Accuracy vs Tree Depth')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison Bar Chart
ax3 = plt.subplot(3, 3, 3)
x_pos = np.arange(len(comparison))
width = 0.35
ax3.bar(x_pos - width/2, comparison['Train Accuracy'], width, label='Train', alpha=0.8)
ax3.bar(x_pos + width/2, comparison['Test Accuracy'], width, label='Test', alpha=0.8)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison['Model'], rotation=15, ha='right')
ax3.set_ylabel('Accuracy')
ax3.set_title('Model Comparison')
ax3.legend()
ax3.set_ylim([0.8, 1.0])

# Plot 4: Confusion Matrix - Decision Tree
ax4 = plt.subplot(3, 3, 4)
cm_dt = confusion_matrix(y_test, y_pred_dt)
im4 = ax4.imshow(cm_dt, cmap='Blues')
ax4.set_xticks(np.arange(len(wine.target_names)))
ax4.set_yticks(np.arange(len(wine.target_names)))
ax4.set_xticklabels(wine.target_names)
ax4.set_yticklabels(wine.target_names)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title('Confusion Matrix: Decision Tree')
for i in range(len(wine.target_names)):
    for j in range(len(wine.target_names)):
        text = ax4.text(j, i, cm_dt[i, j], ha="center", va="center",
                       color="white" if cm_dt[i, j] > cm_dt.max()/2 else "black")
plt.colorbar(im4, ax=ax4)

# Plot 5: Confusion Matrix - Logistic Regression
ax5 = plt.subplot(3, 3, 5)
y_pred_lr = lr_model.predict(X_test_scaled)
cm_lr = confusion_matrix(y_test, y_pred_lr)
im5 = ax5.imshow(cm_lr, cmap='Greens')
ax5.set_xticks(np.arange(len(wine.target_names)))
ax5.set_yticks(np.arange(len(wine.target_names)))
ax5.set_xticklabels(wine.target_names)
ax5.set_yticklabels(wine.target_names)
ax5.set_xlabel('Predicted')
ax5.set_ylabel('Actual')
ax5.set_title('Confusion Matrix: Logistic Regression')
for i in range(len(wine.target_names)):
    for j in range(len(wine.target_names)):
        text = ax5.text(j, i, cm_lr[i, j], ha="center", va="center",
                       color="white" if cm_lr[i, j] > cm_lr.max()/2 else "black")
plt.colorbar(im5, ax=ax5)

# Plot 6: Cross-validation scores
ax6 = plt.subplot(3, 3, 6)
ax6.bar(range(1, 6), cv_scores, color='coral')
ax6.axhline(y=cv_scores.mean(), color='red', linestyle='--', label='Mean')
ax6.set_xlabel('Fold')
ax6.set_ylabel('Accuracy')
ax6.set_title('5-Fold Cross-Validation Scores')
ax6.legend()
ax6.set_ylim([0.8, 1.0])

# Plot 7: Decision Tree Visualization (small tree)
ax7 = plt.subplot(3, 3, (7, 9))
small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
small_tree.fit(X_train, y_train)
plot_tree(small_tree, 
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,
          rounded=True,
          ax=ax7,
          fontsize=8)
ax7.set_title('Decision Tree Visualization (max_depth=3)')

plt.tight_layout()
plt.savefig('D:/ai_engineering/week2_ml_basics/day2_decision_trees_results.png', dpi=150)
print("✅ Visualization saved: day2_decision_trees_results.png")
plt.close()

# ============================================
# STEP 10: Classification Reports
# ============================================
print("\n>>> STEP 10: Detailed Classification Reports")

print("\n--- Decision Tree Report ---")
print(classification_report(y_test, y_pred_dt, target_names=wine.target_names))

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, y_pred_lr, target_names=wine.target_names))

# ============================================
# BONUS: Binary Classification (Cancer Dataset)
# ============================================
print("\n" + "="*70)
print("BONUS: Binary Classification (Breast Cancer Dataset)")
print("="*70)

# Load breast cancer dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print(f"\nDataset: {cancer.data.shape[0]} samples, {cancer.data.shape[1]} features")
print(f"Classes: {cancer.target_names}")

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_cancer = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_cancer.fit(X_train_c, y_train_c)

# Train Logistic Regression
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

lr_cancer = LogisticRegression(max_iter=1000, random_state=42)
lr_cancer.fit(X_train_c_scaled, y_train_c)

# Compare
print("\nDecision Tree Accuracy:", dt_cancer.score(X_test_c, y_test_c))
print("Logistic Regression Accuracy:", lr_cancer.score(X_test_c_scaled, y_test_c))

print("\nDecision Tree Report:")
y_pred_dt_cancer = dt_cancer.predict(X_test_c)
print(classification_report(y_test_c, y_pred_dt_cancer, target_names=cancer.target_names))

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. DECISION TREES:
   - Easy to interpret (tree structure)
   - No feature scaling needed
   - Prone to overfitting without constraints

2. OVERFITTING:
   - Training accuracy >> Test accuracy
   - Control with max_depth, min_samples_split, min_samples_leaf

3. FEATURE IMPORTANCE:
   - Shows which features matter most
   - Useful for feature selection

4. CROSS-VALIDATION:
   - More reliable than single train/test split
   - Use 5-fold or 10-fold CV

5. MODEL COMPARISON:
   - Try multiple algorithms
   - Decision Trees vs Logistic Regression
   - Choose based on test accuracy and interpretability

WHEN TO USE WHAT:
- Decision Trees: When interpretability is important
- Logistic Regression: When you need probability estimates
- Both: Compare and choose the best!
""")
print("="*70)

print("\n✅ Day 2-3 Complete! Move to day3_regression.py")