"""
WEEK 2 - DAY 3: Regression Models
==================================
Learn to predict continuous values with Linear Regression



Topics:
- Linear Regression
- Regression metrics (R², MSE, MAE)
- Feature correlation
- Residual analysis
- Polynomial features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("WEEK 2 - DAY 3: Regression Models")
print("="*70)

# ============================================
# STEP 1: Load California Housing Dataset
# ============================================
print("\n>>> STEP 1: Loading California Housing Dataset")

housing = fetch_california_housing()
X = housing.data
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Target: House prices (in $100,000s)")
print(f"Features: {housing.feature_names}")

# Create DataFrame
df = pd.DataFrame(X, columns=housing.feature_names)
df['MedHouseValue'] = y

print("\nFirst 5 samples:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

# ============================================
# STEP 2: Exploratory Data Analysis
# ============================================
print("\n>>> STEP 2: Exploratory Data Analysis")

# Target distribution
print("\nTarget variable statistics:")
print(f"Mean price: ${y.mean() * 100000:.2f}")
print(f"Median price: ${np.median(y) * 100000:.2f}")
print(f"Min price: ${y.min() * 100000:.2f}")
print(f"Max price: ${y.max() * 100000:.2f}")

# Correlation with target
correlations = df.corr()['MedHouseValue'].sort_values(ascending=False)
print("\nCorrelation with house price:")
print(correlations)

# ============================================
# STEP 3: Feature Selection (top correlated)
# ============================================
print("\n>>> STEP 3: Using Subset of Data for Faster Training")

# Use only 20% of data for faster training (optional)
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.2, random_state=42)

print(f"Using {len(X_subset)} samples for training")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================
# STEP 4: Train Linear Regression
# ============================================
print("\n>>> STEP 4: Training Linear Regression")

# Create and train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("✅ Model trained!")

# Model coefficients
print("\nModel coefficients:")
coef_df = pd.DataFrame({
    'Feature': housing.feature_names,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df)

print(f"\nIntercept: {lr_model.intercept_:.4f}")

# ============================================
# STEP 5: Make Predictions
# ============================================
print("\n>>> STEP 5: Making Predictions")

# Predict on test set
y_pred = lr_model.predict(X_test)

# Show some predictions
comparison = pd.DataFrame({
    'Actual': y_test[:10] * 100000,
    'Predicted': y_pred[:10] * 100000,
    'Difference': (y_test[:10] - y_pred[:10]) * 100000
})
print("\nFirst 10 predictions (in $):")
print(comparison)

# ============================================
# STEP 6: Evaluate Model
# ============================================
print("\n>>> STEP 6: Model Evaluation")

# Calculate metrics
train_score = lr_model.score(X_train, y_train)
test_score = lr_model.score(X_test, y_test)

y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# R² Score (coefficient of determination)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Mean Squared Error
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Root Mean Squared Error
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Mean Absolute Error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("\n--- Regression Metrics ---")
print(f"\nR² Score (1.0 = perfect):")
print(f"  Training: {r2_train:.4f}")
print(f"  Test: {r2_test:.4f}")

print(f"\nRMSE (lower is better, in $100k):")
print(f"  Training: {rmse_train:.4f} (${rmse_train * 100000:.0f})")
print(f"  Test: {rmse_test:.4f} (${rmse_test * 100000:.0f})")

print(f"\nMAE (lower is better, in $100k):")
print(f"  Training: {mae_train:.4f} (${mae_train * 100000:.0f})")
print(f"  Test: {mae_test:.4f} (${mae_test * 100000:.0f})")

# ============================================
# STEP 7: Residual Analysis
# ============================================
print("\n>>> STEP 7: Residual Analysis")

# Calculate residuals (errors)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

print(f"\nResiduals statistics (Test set):")
print(f"Mean: {residuals_test.mean():.6f} (should be close to 0)")
print(f"Std: {residuals_test.std():.4f}")

# ============================================
# STEP 8: Visualizations
# ============================================
print("\n>>> STEP 8: Creating Visualizations")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Actual vs Predicted
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(y_test, y_pred, alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Actual Price ($100k)')
ax1.set_ylabel('Predicted Price ($100k)')
ax1.set_title('Actual vs Predicted Prices')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs Predicted
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(y_pred, residuals_test, alpha=0.5, s=10)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Price ($100k)')
ax2.set_ylabel('Residuals')
ax2.set_title('Residual Plot')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', label='Zero error')
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Residuals')
ax3.legend()

# Plot 4: Feature Coefficients
ax4 = plt.subplot(3, 3, 4)
colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax4.set_xlabel('Coefficient Value')
ax4.set_title('Feature Importance (Coefficients)')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# Plot 5: Metrics Comparison
ax5 = plt.subplot(3, 3, 5)
metrics = ['R² Score', 'RMSE', 'MAE']
train_vals = [r2_train, rmse_train, mae_train]
test_vals = [r2_test, rmse_test, mae_test]
x_pos = np.arange(len(metrics))
width = 0.35
ax5.bar(x_pos - width/2, train_vals, width, label='Train', alpha=0.8)
ax5.bar(x_pos + width/2, test_vals, width, label='Test', alpha=0.8)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(metrics)
ax5.set_ylabel('Value')
ax5.set_title('Metrics Comparison')
ax5.legend()

# Plot 6: Correlation Heatmap (top features)
ax6 = plt.subplot(3, 3, 6)
top_features = ['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'MedHouseValue']
corr_subset = df[top_features].corr()
im = ax6.imshow(corr_subset, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax6.set_xticks(range(len(top_features)))
ax6.set_yticks(range(len(top_features)))
ax6.set_xticklabels(top_features, rotation=45, ha='right')
ax6.set_yticklabels(top_features)
ax6.set_title('Correlation Heatmap')
plt.colorbar(im, ax=ax6)
for i in range(len(top_features)):
    for j in range(len(top_features)):
        text = ax6.text(j, i, f'{corr_subset.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

# Plot 7: Feature vs Target (MedInc)
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(df['MedInc'][:1000], df['MedHouseValue'][:1000], alpha=0.3, s=5)
ax7.set_xlabel('Median Income')
ax7.set_ylabel('House Price ($100k)')
ax7.set_title('Income vs House Price')
ax7.grid(True, alpha=0.3)

# Plot 8: Error distribution
ax8 = plt.subplot(3, 3, 8)
errors = np.abs(residuals_test) * 100000  # Convert to dollars
ax8.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax8.axvline(x=errors.mean(), color='r', linestyle='--', 
           label=f'Mean: ${errors.mean():.0f}')
ax8.set_xlabel('Absolute Error ($)')
ax8.set_ylabel('Frequency')
ax8.set_title('Prediction Error Distribution')
ax8.legend()

# Plot 9: Q-Q plot for residuals normality
ax9 = plt.subplot(3, 3, 9)
from scipy import stats
stats.probplot(residuals_test, dist="norm", plot=ax9)
ax9.set_title('Q-Q Plot (Residuals Normality Check)')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week2_ml_basics/day3_regression_results.png', dpi=150)
print("✅ Visualization saved: day3_regression_results.png")
plt.close()

# ============================================
# BONUS: Polynomial Regression
# ============================================
print("\n" + "="*70)
print("BONUS: Polynomial Regression")
print("="*70)

# Use only 2 features for polynomial example
X_poly = X_train[:, [0, 5]]  # MedInc and Latitude
X_poly_test = X_test[:, [0, 5]]

# Degree 2 polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_transformed = poly.fit_transform(X_poly)
X_poly_test_transformed = poly.transform(X_poly_test)

print(f"\nOriginal features: {X_poly.shape[1]}")
print(f"Polynomial features (degree=2): {X_poly_transformed.shape[1]}")
print(f"Feature names: {poly.get_feature_names_out(['MedInc', 'Latitude'])}")

# Train polynomial model
poly_model = LinearRegression()
poly_model.fit(X_poly_transformed, y_train)

# Evaluate
r2_poly = poly_model.score(X_poly_test_transformed, y_test)
print(f"\nPolynomial Regression R²: {r2_poly:.4f}")
print(f"Linear Regression R² (with 2 features): {LinearRegression().fit(X_poly, y_train).score(X_poly_test, y_test):.4f}")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. LINEAR REGRESSION:
   - Predicts continuous values
   - Assumes linear relationship
   - Fast and interpretable

2. REGRESSION METRICS:
   - R² Score: How much variance is explained (0 to 1, higher is better)
   - RMSE: Average prediction error (lower is better)
   - MAE: Mean absolute error (lower is better)

3. RESIDUALS:
   - Should be randomly distributed around 0
   - Pattern in residuals = model missing something
   - Check normality with Q-Q plot

4. FEATURE IMPORTANCE:
   - Coefficients show feature impact
   - Positive = increases target
   - Negative = decreases target

5. POLYNOMIAL FEATURES:
   - Capture non-linear relationships
   - Increases model complexity
   - Risk of overfitting

INTERPRETATION:
- R² = 0.60 means model explains 60% of price variation
- RMSE = $68k means average error is $68,000
- Use residual plots to diagnose problems
""")
print("="*70)

print("\n✅ Day 4 Complete! Move to day3_clustering.py")