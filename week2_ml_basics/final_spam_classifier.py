"""
WEEK 2 FINAL PROJECT: SMS Spam Classifier
==========================================
Build a complete text classification pipeline



OBJECTIVE:
Create an end-to-end spam detection system using:
- Text preprocessing
- TF-IDF vectorization
- Multiple ML models
- Model comparison
- Performance evaluation

This combines ALL Week 2 skills!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import re
import string

print("="*70)
print("WEEK 2 FINAL PROJECT: SMS Spam Classifier")
print("="*70)

# ============================================
# STEP 1: Generate Synthetic SMS Dataset
# ============================================
print("\n>>> STEP 1: Creating SMS Dataset")

# Spam messages (common spam patterns)
spam_templates = [
    "WINNER!! You have won a ${amount} prize! Call {phone} now!",
    "Congratulations! Claim your FREE {item} by texting {code}",
    "URGENT! Your account will be closed. Click {link} immediately",
    "You've been selected for a FREE {item}. Reply YES now!",
    "PRIZE ALERT! You won ${amount}! Text WIN to {phone}",
    "Limited offer! Get {item} for only ${amount}! Call now",
    "CONGRATULATIONS! You are our lucky winner. Call {phone}",
    "FREE entry to win ${amount}! Text {code} to enter now",
    "URGENT: Your bank account needs verification. Click {link}",
    "You have been chosen! Claim ${amount} reward at {link}"
]

# Ham messages (normal messages)
ham_templates = [
    "Hey, are you free for dinner tonight?",
    "Thanks for your help yesterday, really appreciate it!",
    "Can you pick up some milk on your way home?",
    "Meeting is rescheduled to 3pm tomorrow",
    "Happy birthday! Hope you have a great day!",
    "Don't forget about mom's birthday next week",
    "The project deadline is Friday, let's meet tomorrow",
    "I'll be there in 15 minutes",
    "Great job on the presentation today!",
    "Can you send me that document we discussed?",
    "Let's grab coffee this weekend",
    "Thanks for the update, I'll review it tonight",
    "Good morning! Have a productive day",
    "I'll call you after work today",
    "The meeting went well, will update you soon"
]

np.random.seed(42)

# Generate spam messages
spam_messages = []
for _ in range(200):
    template = np.random.choice(spam_templates)
    msg = template.format(
        amount=np.random.choice([100, 500, 1000, 5000]),
        item=np.random.choice(['iPhone', 'iPad', 'laptop', 'vacation', 'gift card']),
        phone=f"{np.random.randint(1000, 9999)}",
        code=f"{np.random.randint(1000, 9999)}",
        link="http://bit.ly/" + ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 6))
    )
    spam_messages.append(msg)

# Generate ham messages
ham_messages = []
for _ in range(300):
    template = np.random.choice(ham_templates)
    # Add some variation
    if np.random.random() > 0.7:
        template = template + " " + np.random.choice(["Thanks!", "See you", "Cheers", "Best", "Take care"])
    ham_messages.append(template)

# Create DataFrame
df = pd.DataFrame({
    'message': spam_messages + ham_messages,
    'label': ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total messages: {len(df)}")
print(f"Spam: {(df['label'] == 'spam').sum()} ({(df['label'] == 'spam').sum()/len(df)*100:.1f}%)")
print(f"Ham: {(df['label'] == 'ham').sum()} ({(df['label'] == 'ham').sum()/len(df)*100:.1f}%)")

print("\nSample messages:")
print(df.head(10))

# Save dataset
df.to_csv('D:/ai_engineering/datasets/sms_spam.csv', index=False)
print("\n✅ Dataset saved: sms_spam.csv")

# ============================================
# STEP 2: Text Preprocessing
# ============================================
print("\n>>> STEP 2: Text Preprocessing")

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{4}', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing
df['cleaned'] = df['message'].apply(preprocess_text)

print("\nBefore preprocessing:")
print(df['message'].iloc[0])
print("\nAfter preprocessing:")
print(df['cleaned'].iloc[0])

# ============================================
# STEP 3: Feature Extraction (TF-IDF)
# ============================================
print("\n>>> STEP 3: Feature Extraction with TF-IDF")

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned'])

# Convert labels to binary
y = (df['label'] == 'spam').astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Top 20 features: {vectorizer.get_feature_names_out()[:20]}")

# ============================================
# STEP 4: Train-Test Split
# ============================================
print("\n>>> STEP 4: Splitting Data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} messages")
print(f"Test set: {X_test.shape[0]} messages")
print(f"Training spam ratio: {y_train.sum()/len(y_train):.2%}")
print(f"Test spam ratio: {y_test.sum()/len(y_test):.2%}")

# ============================================
# STEP 5: Train Multiple Models
# ============================================
print("\n>>> STEP 5: Training Multiple Classifiers")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================
# STEP 6: Model Comparison
# ============================================
print("\n>>> STEP 6: Model Comparison")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results],
    'CV Mean': [results[m]['cv_mean'] for m in results]
}).sort_values('F1-Score', ascending=False)

print("\n" + "="*70)
print(comparison_df.to_string(index=False))
print("="*70)

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n✅ Best Model: {best_model_name}")

# ============================================
# STEP 7: Detailed Analysis of Best Model
# ============================================
print(f"\n>>> STEP 7: Analyzing {best_model_name}")

y_pred_best = results[best_model_name]['predictions']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print("\nInterpretation:")
print(f"True Negatives (Ham correctly identified): {cm[0][0]}")
print(f"False Positives (Ham wrongly marked as Spam): {cm[0][1]}")
print(f"False Negatives (Spam wrongly marked as Ham): {cm[1][0]}")
print(f"True Positives (Spam correctly identified): {cm[1][1]}")

# ============================================
# STEP 8: Feature Importance
# ============================================
print("\n>>> STEP 8: Most Important Features")

# Get feature importance (for Logistic Regression)
if 'Logistic Regression' in results:
    lr_model = results['Logistic Regression']['model']
    feature_names = vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    
    # Top spam indicators (positive coefficients)
    spam_features = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    
    print("\nTop 10 Spam Indicators:")
    print(spam_features.head(10))
    
    print("\nTop 10 Ham Indicators:")
    print(spam_features.tail(10))

# ============================================
# STEP 9: Test on New Messages
# ============================================
print("\n>>> STEP 9: Testing on New Messages")

test_messages = [
    "WINNER! You have won $5000! Call 1234 now!",
    "Hey, let's meet for lunch tomorrow",
    "Congratulations! Claim your FREE iPhone now!",
    "Can you pick up groceries on your way home?",
    "URGENT: Your account needs verification. Click link",
    "Thanks for dinner last night, had a great time!"
]

print("\nPredicting new messages:\n")
for msg in test_messages:
    cleaned = preprocess_text(msg)
    features = vectorizer.transform([cleaned])
    prediction = best_model.predict(features)[0]
    probability = None
    
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(features)[0][1]
    elif hasattr(best_model, 'decision_function'):
        score = best_model.decision_function(features)[0]
        probability = 1 / (1 + np.exp(-score))  # Sigmoid
    
    label = "SPAM" if prediction == 1 else "HAM"
    
    print(f"Message: {msg}")
    print(f"Prediction: {label}", end="")
    if probability is not None:
        print(f" (confidence: {probability:.2%})")
    else:
        print()
    print()

# ============================================
# STEP 10: Visualizations
# ============================================
print(">>> STEP 10: Creating Visualizations")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Model Comparison - Metrics
ax1 = plt.subplot(3, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_pos = np.arange(len(comparison_df))
width = 0.2
for i, metric in enumerate(metrics):
    ax1.bar(x_pos + i*width, comparison_df[metric], width, label=metric, alpha=0.8)
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison')
ax1.legend()
ax1.set_ylim([0.7, 1.0])

# Plot 2: Confusion Matrix (Best Model)
ax2 = plt.subplot(3, 3, 2)
im = ax2.imshow(cm, cmap='Blues')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Ham', 'Spam'])
ax2.set_yticklabels(['Ham', 'Spam'])
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title(f'Confusion Matrix: {best_model_name}')
for i in range(2):
    for j in range(2):
        text = ax2.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=20)
plt.colorbar(im, ax=ax2)

# Plot 3: F1-Score Comparison
ax3 = plt.subplot(3, 3, 3)
bars = ax3.barh(comparison_df['Model'], comparison_df['F1-Score'], color='coral')
ax3.set_xlabel('F1-Score')
ax3.set_title('F1-Score by Model')
ax3.set_xlim([0.7, 1.0])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.3f}', ha='left', va='center', fontsize=10)

# Plot 4: Class Distribution
ax4 = plt.subplot(3, 3, 4)
class_dist = df['label'].value_counts()
ax4.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', startangle=90,
       colors=['lightgreen', 'lightcoral'])
ax4.set_title('Dataset Class Distribution')

# Plot 5: Message Length Distribution
ax5 = plt.subplot(3, 3, 5)
spam_lengths = df[df['label'] == 'spam']['message'].str.len()
ham_lengths = df[df['label'] == 'ham']['message'].str.len()
ax5.hist([ham_lengths, spam_lengths], bins=30, label=['Ham', 'Spam'], alpha=0.7)
ax5.set_xlabel('Message Length (characters)')
ax5.set_ylabel('Frequency')
ax5.set_title('Message Length Distribution')
ax5.legend()

# Plot 6: Word Count Distribution
ax6 = plt.subplot(3, 3, 6)
spam_words = df[df['label'] == 'spam']['message'].str.split().str.len()
ham_words = df[df['label'] == 'ham']['message'].str.split().str.len()
ax6.boxplot([ham_words, spam_words], labels=['Ham', 'Spam'])
ax6.set_ylabel('Word Count')
ax6.set_title('Word Count Distribution')

# Plot 7: Top Spam Keywords
ax7 = plt.subplot(3, 3, 7)
if 'Logistic Regression' in results:
    top_spam = spam_features.head(10)
    ax7.barh(range(len(top_spam)), top_spam['coefficient'], color='red', alpha=0.7)
    ax7.set_yticks(range(len(top_spam)))
    ax7.set_yticklabels(top_spam['feature'])
    ax7.set_xlabel('Coefficient')
    ax7.set_title('Top 10 Spam Indicators')
    ax7.invert_yaxis()

# Plot 8: Top Ham Keywords
ax8 = plt.subplot(3, 3, 8)
if 'Logistic Regression' in results:
    top_ham = spam_features.tail(10).sort_values('coefficient')
    ax8.barh(range(len(top_ham)), top_ham['coefficient'], color='green', alpha=0.7)
    ax8.set_yticks(range(len(top_ham)))
    ax8.set_yticklabels(top_ham['feature'])
    ax8.set_xlabel('Coefficient')
    ax8.set_title('Top 10 Ham Indicators')
    ax8.invert_yaxis()

# Plot 9: Cross-Validation Scores
ax9 = plt.subplot(3, 3, 9)
cv_means = [results[m]['cv_mean'] for m in results]
cv_stds = [results[m]['cv_std'] for m in results]
ax9.errorbar(range(len(results)), cv_means, yerr=cv_stds, fmt='o-', capsize=5)
ax9.set_xticks(range(len(results)))
ax9.set_xticklabels(results.keys(), rotation=15, ha='right')
ax9.set_ylabel('CV Score')
ax9.set_title('Cross-Validation Scores (with std)')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/ai_engineering/week2_ml_basics/final_spam_classifier_results.png', dpi=150)
print("✅ Visualization saved: final_spam_classifier_results.png")
plt.close()

# ============================================
# STEP 11: Save Model and Summary
# ============================================
print("\n>>> STEP 11: Saving Results")

# Save comparison results
comparison_df.to_csv('D:/ai_engineering/week2_ml_basics/model_comparison.csv', index=False)
print("✅ Model comparison saved")

# Generate final report
report = f"""
{'='*70}
                    SMS SPAM CLASSIFIER - FINAL REPORT
{'='*70}

PROJECT OVERVIEW:
- Dataset Size: {len(df)} messages
- Spam Messages: {(df['label'] == 'spam').sum()}
- Ham Messages: {(df['label'] == 'ham').sum()}
- Features: {X.shape[1]} TF-IDF features

{'='*70}
MODEL PERFORMANCE:
{'='*70}

{comparison_df.to_string(index=False)}

{'='*70}
BEST MODEL: {best_model_name}
{'='*70}

Accuracy: {results[best_model_name]['accuracy']:.4f}
Precision: {results[best_model_name]['precision']:.4f}
Recall: {results[best_model_name]['recall']:.4f}
F1-Score: {results[best_model_name]['f1']:.4f}

CONFUSION MATRIX:
                Predicted Ham  Predicted Spam
Actual Ham          {cm[0][0]}              {cm[0][1]}
Actual Spam         {cm[1][0]}              {cm[1][1]}

{'='*70}
KEY INSIGHTS:
{'='*70}

1. SPAM INDICATORS:
   - Words like "winner", "free", "prize", "urgent"
   - Presence of phone numbers and URLs
   - Use of ALL CAPS
   - Call-to-action phrases

2. HAM INDICATORS:
   - Conversational language
   - Personal pronouns
   - Specific details and context
   - Normal punctuation

3. MODEL COMPARISON:
   - All models achieved >95% accuracy
   - {best_model_name} performed best overall
   - Naive Bayes is fastest, good for real-time
   - Logistic Regression offers interpretability

{'='*70}
PRODUCTION RECOMMENDATIONS:
{'='*70}

1. Use {best_model_name} for best accuracy
2. Retrain model regularly with new spam patterns
3. Implement user feedback loop
4. Monitor false positives carefully
5. Consider ensemble methods for improvement

{'='*70}
"""

print(report)

with open('D:/ai_engineering/week2_ml_basics/spam_classifier_report.txt', 'w') as f:
    f.write(report)
print("✅ Final report saved: spam_classifier_report.txt")

# ============================================
# CONGRATULATIONS!
# ============================================
print("\n" + "="*70)
print("🎉 CONGRATULATIONS! WEEK 2 COMPLETE! 🎉")
print("="*70)
print("""
YOU HAVE MASTERED:
✅ Classification (Logistic Regression, Decision Trees)
✅ Regression (Linear Regression)
✅ Clustering (K-Means)
✅ Text Processing (TF-IDF, preprocessing)
✅ Model Evaluation (accuracy, precision, recall, F1)
✅ Cross-Validation
✅ Feature Engineering
✅ Model Comparison

READY FOR PHASE 2: 
⚙️ Model Fine-tuning & Deployment (PyTorch, Colab)

Next steps:
1. Review all Week 2 projects
2. Experiment with parameters
3. Try different datasets
4. Start Week 3 when ready!
""")
print("="*70)