"""
WEEK 2 - DAY 1: Classification Basics

Learn supervised learning with Logistic Regression


Topics:
- Train/test split
- Logistic Regression
- Model evaluation
- Confusion matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

print("="*70)
print("WEEK 2 - DAY 1: Classification Basics")
print("="*70)

# ============================================
# STEP 1: Load and Explore Dataset
# ============================================
print("\n>>> STEP 1: Loading Iris Dataset")

# Load dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: 0=setosa, 1=versicolor, 2=virginica

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")

# Create DataFrame for exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\nFirst 5 samples:")
print(df.head())

print("\nClass distribution:")
print(df['species_name'].value_counts())

# ============================================
# STEP 2: Train-Test Split
# ============================================
print("\n>>> STEP 2: Splitting Data")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training classes distribution: {np.bincount(y_train)}")
print(f"Test classes distribution: {np.bincount(y_test)}")

# ============================================
# STEP 3: Feature Scaling (Optional but recommended)
# ============================================
print("\n>>> STEP 3: Feature Scaling")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled (mean=0, std=1)")
print(f"Training mean: {X_train_scaled.mean(axis=0).round(2)}")
print(f"Training std: {X_train_scaled.std(axis=0).round(2)}")

# ============================================
# STEP 4: Train Logistic Regression Model
# ============================================
print("\n>>> STEP 4: Training Logistic Regression")

# Create and train model
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Model trained successfully!")

# ============================================
# STEP 5: Make Predictions
# ============================================
print("\n>>> STEP 5: Making Predictions")

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Show some predictions
print("\nFirst 10 predictions vs actual:")
comparison = pd.DataFrame({
    'Actual': [iris.target_names[i] for i in y_test[:10]],
    'Predicted': [iris.target_names[i] for i in y_pred[:10]]
})
print(comparison)

# ============================================
# STEP 6: Evaluate Model
# ============================================
print("\n>>> STEP 6: Model Evaluation")

# Calculate accuracy
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nExplanation:")
print("Rows = Actual class, Columns = Predicted class")
print("Diagonal values = Correct predictions")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ============================================
# STEP 7: Visualize Results
# ============================================
print("\n>>> STEP 7: Creating Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confusion Matrix Heatmap
ax1 = axes[0, 0]
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks(np.arange(len(iris.target_names)))
ax1.set_yticks(np.arange(len(iris.target_names)))
ax1.set_xticklabels(iris.target_names)
ax1.set_yticklabels(iris.target_names)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

# Add text annotations
for i in range(len(iris.target_names)):
    for j in range(len(iris.target_names)):
        text = ax1.text(j, i, cm[i, j], ha="center", va="center", 
                       color="white" if cm[i, j] > cm.max()/2 else "black")

plt.colorbar(im, ax=ax1)

# Plot 2: Feature Importance (coefficients)
ax2 = axes[0, 1]
# For multi-class, show coefficients for first class
coef = model.coef_[0]
features = iris.feature_names
colors = ['red' if c < 0 else 'green' for c in coef]
ax2.barh(features, coef, color=colors)
ax2.set_xlabel('Coefficient Value')
ax2.set_title('Feature Importance (Class 0: Setosa)')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# Plot 3: Scatter plot with predictions
ax3 = axes[1, 0]
# Use first 2 features for visualization
correct = y_test == y_pred
ax3.scatter(X_test[correct, 0], X_test[correct, 1], 
           c=y_test[correct], cmap='viridis', marker='o', 
           s=100, alpha=0.7, label='Correct', edgecolors='black')
ax3.scatter(X_test[~correct, 0], X_test[~correct, 1], 
           c='red', marker='x', s=200, alpha=1.0, 
           label='Wrong', linewidths=3)
ax3.set_xlabel(iris.feature_names[0])
ax3.set_ylabel(iris.feature_names[1])
ax3.set_title('Predictions on Test Set')
ax3.legend()

# Plot 4: Accuracy Comparison
ax4 = axes[1, 1]
categories = ['Training', 'Test']
accuracies = [train_accuracy * 100, test_accuracy * 100]
bars = ax4.bar(categories, accuracies, color=['blue', 'orange'])
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Model Accuracy Comparison')
ax4.set_ylim([0, 105])
ax4.axhline(y=100, color='green', linestyle='--', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week2_ml_basics/day1_classification_results.png', dpi=150)
print("✅ Visualization saved: day1_classification_results.png")
plt.close()

# ============================================
# EXERCISE: Predict New Samples
# ============================================
print("\n>>> EXERCISE: Predicting New Flowers")

# Create some hypothetical flowers
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Looks like Setosa
    [6.5, 3.0, 5.5, 1.8],  # Looks like Virginica
    [5.7, 2.8, 4.1, 1.3]   # Looks like Versicolor
])

# Scale the new data
new_flowers_scaled = scaler.transform(new_flowers)

# Predict
predictions = model.predict(new_flowers_scaled)
probabilities = model.predict_proba(new_flowers_scaled)

print("\nNew flower predictions:")
for i, (flower, pred, proba) in enumerate(zip(new_flowers, predictions, probabilities)):
    print(f"\nFlower {i+1}: {flower}")
    print(f"Predicted: {iris.target_names[pred]}")
    print(f"Probabilities: Setosa={proba[0]:.3f}, Versicolor={proba[1]:.3f}, Virginica={proba[2]:.3f}")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. TRAIN-TEST SPLIT: Always split data to evaluate on unseen data
2. FEATURE SCALING: Standardize features for better performance
3. ACCURACY: Percentage of correct predictions
4. CONFUSION MATRIX: Shows where model makes mistakes
5. CLASSIFICATION REPORT: Precision, Recall, F1-score per class

NEXT STEPS:
- Try with different test_size (0.3, 0.4)
- Experiment with different random_state
- Try without feature scaling and compare results
- Use only 2 features and visualize decision boundary
""")
print("="*70)

# ============================================
# CHALLENGE: Try Binary Classification
# ============================================
print("\n>>> CHALLENGE: Binary Classification (Setosa vs Others)")

# Create binary target: 0 = Setosa, 1 = Others
y_binary = (y != 0).astype(int)

# Split data
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Scale
X_train_b_scaled = scaler.fit_transform(X_train_b)
X_test_b_scaled = scaler.transform(X_test_b)

# Train
model_binary = LogisticRegression(max_iter=200, random_state=42)
model_binary.fit(X_train_b_scaled, y_train_b)

# Evaluate
accuracy_binary = model_binary.score(X_test_b_scaled, y_test_b)
print(f"\nBinary Classification Accuracy: {accuracy_binary:.4f}")

y_pred_binary = model_binary.predict(X_test_b_scaled)
print("\nBinary Confusion Matrix:")
print(confusion_matrix(y_test_b, y_pred_binary))
print("\nBinary Classification Report:")
print(classification_report(y_test_b, y_pred_binary, 
                           target_names=['Setosa', 'Others']))

print("\n✅ Day 1 Complete! Move to day2_decision_trees.py")