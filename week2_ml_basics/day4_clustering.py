"""
WEEK 2 - DAY 5: Clustering (Unsupervised Learning)
===================================================
Learn K-Means clustering to group similar data


Topics:
- K-Means clustering
- Elbow method
- Silhouette score
- Cluster visualization
- Customer segmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WEEK 2 - DAY 5: Clustering with K-Means")
print("="*70)

# ============================================
# STEP 1: Generate Synthetic Customer Data
# ============================================
print("\n>>> STEP 1: Generating Customer Data")

np.random.seed(42)

# Create synthetic customer data
n_customers = 300

# Generate features
annual_income = np.random.normal(60, 20, n_customers)  # in $1000s
annual_income = np.clip(annual_income, 15, 140)

spending_score = np.random.normal(50, 25, n_customers)  # 1-100
spending_score = np.clip(spending_score, 1, 100)

age = np.random.normal(40, 15, n_customers)
age = np.clip(age, 18, 70)

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': [f'C{i:04d}' for i in range(1, n_customers + 1)],
    'Age': age.astype(int),
    'Annual_Income': annual_income.round(1),
    'Spending_Score': spending_score.round(0).astype(int)
})

print(f"Generated {len(df)} customer records")
print("\nFirst 5 customers:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

# Save data
df.to_csv('D:/ai_engineering/datasets/customers.csv', index=False)
print("\n✅ Data saved: customers.csv")

# ============================================
# STEP 2: Exploratory Data Analysis
# ============================================
print("\n>>> STEP 2: Exploratory Data Analysis")

print("\nFeature correlations:")
print(df[['Age', 'Annual_Income', 'Spending_Score']].corr())

# ============================================
# STEP 3: Prepare Data for Clustering
# ============================================
print("\n>>> STEP 3: Data Preparation")

# Select features for clustering
X = df[['Annual_Income', 'Spending_Score']].values

print(f"Features selected: Annual_Income, Spending_Score")
print(f"Data shape: {X.shape}")

# Feature scaling (important for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Features scaled (mean=0, std=1)")

# ============================================
# STEP 4: Elbow Method (Find Optimal K)
# ============================================
print("\n>>> STEP 4: Finding Optimal Number of Clusters (Elbow Method)")

# Try different values of k
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
print("\nK | Inertia | Silhouette Score")
print("-" * 35)
for k, inertia, silh in zip(k_range, inertias, silhouette_scores):
    print(f"{k:2d} | {inertia:7.2f} | {silh:.4f}")

# Find elbow point (where decrease slows down)
# Simple heuristic: largest percentage drop
pct_changes = [100 * (inertias[i] - inertias[i+1]) / inertias[i] 
               for i in range(len(inertias)-1)]
optimal_k_elbow = k_range[np.argmax(pct_changes) + 1]
optimal_k_silh = k_range[np.argmax(silhouette_scores)]

print(f"\n✅ Elbow method suggests: k={optimal_k_elbow}")
print(f"✅ Silhouette method suggests: k={optimal_k_silh}")

# We'll use k=5 for customer segmentation
optimal_k = 5

# ============================================
# STEP 5: Train K-Means with Optimal K
# ============================================
print(f"\n>>> STEP 5: Training K-Means with k={optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to dataframe
df['Cluster'] = clusters

print(f"✅ Clustering complete!")
print(f"\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

# Cluster centers (in original scale)
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

print("\nCluster centers (original scale):")
centers_df = pd.DataFrame(centers_original, 
                         columns=['Annual_Income', 'Spending_Score'])
centers_df.index = [f'Cluster {i}' for i in range(optimal_k)]
print(centers_df)

# ============================================
# STEP 6: Analyze Clusters
# ============================================
print("\n>>> STEP 6: Cluster Analysis")

# Statistics per cluster
cluster_stats = df.groupby('Cluster').agg({
    'Age': ['mean', 'min', 'max'],
    'Annual_Income': ['mean', 'min', 'max'],
    'Spending_Score': ['mean', 'min', 'max']
}).round(1)

print("\nCluster statistics:")
print(cluster_stats)

# Interpret clusters
print("\n--- Cluster Interpretations ---")
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    avg_income = cluster_data['Annual_Income'].mean()
    avg_spending = cluster_data['Spending_Score'].mean()
    
    print(f"\nCluster {i} ({len(cluster_data)} customers):")
    print(f"  Average Income: ${avg_income:.1f}k")
    print(f"  Average Spending Score: {avg_spending:.0f}")
    
    # Label cluster
    if avg_income > 70 and avg_spending > 60:
        label = "High Income, High Spenders (VIP)"
    elif avg_income > 70 and avg_spending < 40:
        label = "High Income, Low Spenders (Savers)"
    elif avg_income < 40 and avg_spending > 60:
        label = "Low Income, High Spenders (Impulsive)"
    elif avg_income < 40 and avg_spending < 40:
        label = "Low Income, Low Spenders (Budget)"
    else:
        label = "Middle Income, Moderate Spenders"
    
    print(f"  Segment: {label}")

# ============================================
# STEP 7: Visualizations
# ============================================
print("\n>>> STEP 7: Creating Visualizations")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Elbow Curve
ax1 = plt.subplot(3, 3, 1)
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Chosen k={optimal_k}')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
ax1.set_title('Elbow Method')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Silhouette Scores
ax2 = plt.subplot(3, 3, 2)
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Chosen k={optimal_k}')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs k')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Clusters Visualization
ax3 = plt.subplot(3, 3, 3)
scatter = ax3.scatter(df['Annual_Income'], df['Spending_Score'], 
                     c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
ax3.scatter(centers_original[:, 0], centers_original[:, 1], 
           c='red', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Centroids')
ax3.set_xlabel('Annual Income ($1000s)')
ax3.set_ylabel('Spending Score (1-100)')
ax3.set_title(f'Customer Segments (k={optimal_k})')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Cluster')

# Plot 4: 3D Visualization with Age
ax4 = plt.subplot(3, 3, 4, projection='3d')
ax4.scatter(df['Annual_Income'], df['Spending_Score'], df['Age'],
           c=df['Cluster'], cmap='viridis', s=30, alpha=0.6)
ax4.set_xlabel('Annual Income')
ax4.set_ylabel('Spending Score')
ax4.set_zlabel('Age')
ax4.set_title('3D Cluster Visualization')

# Plot 5: Cluster Size Distribution
ax5 = plt.subplot(3, 3, 5)
cluster_counts = df['Cluster'].value_counts().sort_index()
bars = ax5.bar(cluster_counts.index, cluster_counts.values, 
              color=['C0', 'C1', 'C2', 'C3', 'C4'])
ax5.set_xlabel('Cluster')
ax5.set_ylabel('Number of Customers')
ax5.set_title('Cluster Size Distribution')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

# Plot 6: Average Income by Cluster
ax6 = plt.subplot(3, 3, 6)
avg_income = df.groupby('Cluster')['Annual_Income'].mean().sort_index()
ax6.bar(avg_income.index, avg_income.values, color='skyblue')
ax6.set_xlabel('Cluster')
ax6.set_ylabel('Average Annual Income ($1000s)')
ax6.set_title('Average Income by Cluster')
ax6.axhline(y=df['Annual_Income'].mean(), color='r', linestyle='--', 
           label='Overall Avg')
ax6.legend()

# Plot 7: Average Spending by Cluster
ax7 = plt.subplot(3, 3, 7)
avg_spending = df.groupby('Cluster')['Spending_Score'].mean().sort_index()
ax7.bar(avg_spending.index, avg_spending.values, color='lightcoral')
ax7.set_xlabel('Cluster')
ax7.set_ylabel('Average Spending Score')
ax7.set_title('Average Spending Score by Cluster')
ax7.axhline(y=df['Spending_Score'].mean(), color='r', linestyle='--',
           label='Overall Avg')
ax7.legend()

# Plot 8: Age Distribution by Cluster
ax8 = plt.subplot(3, 3, 8)
for cluster in range(optimal_k):
    cluster_ages = df[df['Cluster'] == cluster]['Age']
    ax8.hist(cluster_ages, alpha=0.5, label=f'Cluster {cluster}', bins=15)
ax8.set_xlabel('Age')
ax8.set_ylabel('Frequency')
ax8.set_title('Age Distribution by Cluster')
ax8.legend()

# Plot 9: Cluster Characteristics Heatmap
ax9 = plt.subplot(3, 3, 9)
cluster_means = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean()
cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
im = ax9.imshow(cluster_means_normalized.T, cmap='YlOrRd', aspect='auto')
ax9.set_xticks(range(optimal_k))
ax9.set_yticks(range(3))
ax9.set_xticklabels([f'C{i}' for i in range(optimal_k)])
ax9.set_yticklabels(['Age', 'Income', 'Spending'])
ax9.set_title('Normalized Cluster Characteristics')
plt.colorbar(im, ax=ax9)
for i in range(3):
    for j in range(optimal_k):
        text = ax9.text(j, i, f'{cluster_means_normalized.iloc[j, i]:.2f}',
                       ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig('D:/ai_engineering/week2_ml_basics/day4_clustering_results.png', dpi=150)
print("✅ Visualization saved: day4_clustering_results.png")
plt.close()

# ============================================
# STEP 8: Save Results
# ============================================
print("\n>>> STEP 8: Saving Results")

# Save clustered data
df.to_csv('D:/ai_engineering/datasets/customers_clustered.csv', index=False)
print("✅ Clustered data saved: customers_clustered.csv")

# Generate marketing report
report = f"""
{'='*70}
                    CUSTOMER SEGMENTATION REPORT
{'='*70}

CLUSTERING METHOD: K-Means
NUMBER OF CLUSTERS: {optimal_k}
TOTAL CUSTOMERS: {len(df)}

{'='*70}
CLUSTER BREAKDOWN:
{'='*70}

"""

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    report += f"\nCLUSTER {i}: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)\n"
    report += f"  Average Age: {cluster_data['Age'].mean():.1f} years\n"
    report += f"  Average Income: ${cluster_data['Annual_Income'].mean():.1f}k\n"
    report += f"  Average Spending Score: {cluster_data['Spending_Score'].mean():.1f}\n"
    report += "-" * 70 + "\n"

report += f"""
{'='*70}
MARKETING RECOMMENDATIONS:
{'='*70}

1. HIGH-VALUE CUSTOMERS (High Income + High Spending):
   - Premium loyalty programs
   - Exclusive products/services
   - Personalized experiences

2. POTENTIAL CUSTOMERS (High Income + Low Spending):
   - Targeted promotions
   - Upselling campaigns
   - Engagement initiatives

3. BUDGET CONSCIOUS (Low Income + Low Spending):
   - Value deals and discounts
   - Entry-level products
   - Payment plans

4. IMPULSIVE BUYERS (Low Income + High Spending):
   - Flash sales
   - Limited-time offers
   - Credit facilities

{'='*70}
"""

print(report)

with open('D:/ai_engineering/week2_ml_basics/clustering_report.txt', 'w') as f:
    f.write(report)
print("✅ Report saved: clustering_report.txt")

# ============================================
# BONUS: Silhouette Analysis
# ============================================
print("\n" + "="*70)
print("BONUS: Detailed Silhouette Analysis")
print("="*70)

silhouette_avg = silhouette_score(X_scaled, clusters)
sample_silhouette_values = silhouette_samples(X_scaled, clusters)

print(f"\nOverall Silhouette Score: {silhouette_avg:.4f}")
print("(Ranges from -1 to 1; higher is better)")
print("  > 0.7: Strong cluster structure")
print("  0.5-0.7: Medium cluster structure")
print("  < 0.5: Weak cluster structure")

print("\nSilhouette score by cluster:")
for i in range(optimal_k):
    cluster_silhouette_values = sample_silhouette_values[clusters == i]
    print(f"  Cluster {i}: {cluster_silhouette_values.mean():.4f}")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. K-MEANS CLUSTERING:
   - Unsupervised learning (no labels needed)
   - Groups similar data points together
   - Requires specifying k (number of clusters)

2. FINDING OPTIMAL K:
   - Elbow Method: Look for "elbow" in inertia plot
   - Silhouette Score: Higher is better (max 1.0)
   - Domain knowledge helps choose k

3. FEATURE SCALING:
   - CRITICAL for K-Means!
   - Use StandardScaler before clustering
   - Ensures all features contribute equally

4. APPLICATIONS:
   - Customer segmentation
   - Market basket analysis
   - Image compression
   - Anomaly detection

5. LIMITATIONS:
   - Assumes spherical clusters
   - Sensitive to initialization
   - Requires k to be specified

BUSINESS VALUE:
- Targeted marketing campaigns
- Personalized recommendations
- Resource allocation
- Customer retention strategies
""")
print("="*70)

print("\n✅ Day 5 Complete! Ready for Week 2 Final Project!")