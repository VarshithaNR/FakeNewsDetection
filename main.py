import pandas as pd          # Used to work with data like excel tables
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# ===============================
# LOAD DATASET
# ===============================
fake = pd.read_csv("Dataset/fake.csv")
true = pd.read_csv("Dataset/true.csv")

# Add labels
fake["label"] = 0   # 0 = Fake
true["label"] = 1   # 1 = Real

# Combine both
data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Show first 5 rows
print(data.head())

# Check shape (rows, columns)
print("Shape:", data.shape)

# Check column names
print("Columns:", data.columns)

# Dataset info
print("Data Info:")
data.info()

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# ===============================
# EXPLORATORY DATA ANALYSIS (EDA)
# ===============================

# Label Distribution
plt.figure()
data["label"].value_counts().plot(kind="bar", color=["#E24B4A", "#639922"])
plt.xticks([0, 1], ["Fake", "Real"], rotation=0)
plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Article Length Distribution
data["text_len"] = data["text"].fillna("").apply(lambda x: len(x.split()))

plt.figure()
plt.hist(data[data["label"] == 0]["text_len"].clip(upper=1500), bins=50,
         alpha=0.6, color="#E24B4A", label="Fake")
plt.hist(data[data["label"] == 1]["text_len"].clip(upper=1500), bins=50,
         alpha=0.6, color="#639922", label="Real")
plt.title("Article Length Distribution (Words)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Subject Distribution (Fake News)
plt.figure()
fake["subject"].value_counts().plot(kind="bar", color="#E24B4A")
plt.title("Fake News - Subject Distribution")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Subject Distribution (Real News)
plt.figure()
true["subject"].value_counts().plot(kind="bar", color="#639922")
plt.title("Real News - Subject Distribution")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Top Words in Fake vs Real
STOP = set(["the","a","an","and","or","but","in","on","at","to","for","of",
            "with","by","from","is","are","was","were","be","been","have",
            "has","had","do","does","did","will","would","could","should",
            "it","its","this","that","i","we","you","he","she","they",
            "me","us","him","her","them","my","our","your","his","their",
            "what","which","who","as","also","not","no","all","so","just","said"])

def top_words(series, n=15):
    words = []
    for text in series.fillna(""):
        words.extend([w for w in re.sub(r"[^a-z\s]", "", text.lower()).split()
                      if w not in STOP and len(w) > 3])
    return Counter(words).most_common(n)

fake_top = top_words(fake["text"])
true_top = top_words(true["text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
words_f, freq_f = zip(*fake_top)
axes[0].barh(words_f[::-1], freq_f[::-1], color="#E24B4A", alpha=0.85)
axes[0].set_title("Top Words — Fake News")
axes[0].set_xlabel("Frequency")

words_t, freq_t = zip(*true_top)
axes[1].barh(words_t[::-1], freq_t[::-1], color="#639922", alpha=0.85)
axes[1].set_title("Top Words — Real News")
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()

# ===============================
# TEXT PREPROCESSING
# ===============================
def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)          # Remove special characters
    words = [w for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(words)

# Combine title + text, then clean
data["clean_text"] = (data["title"].fillna("") + " " + data["text"].fillna("")).apply(preprocess)
print("Sample cleaned text:")
print(data["clean_text"].iloc[0][:200])

# ===============================
# TF-IDF VECTORISATION
# ===============================
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

X = tfidf.fit_transform(data["clean_text"])
y = data["label"]

print("Vocabulary size:", len(tfidf.vocabulary_))
print("Feature matrix shape:", X.shape)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ===============================
# MODEL 1 — LOGISTIC REGRESSION
# ===============================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create model
lr = LogisticRegression(max_iter=1000, C=1.0)
# Train model
lr.fit(X_train, y_train)
# Predict
y_pred_lr = lr.predict(X_test)
# Evaluate
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=["Fake", "Real"]))

# ===============================
# MODEL 2 — NAIVE BAYES
# ===============================
from sklearn.naive_bayes import MultinomialNB

# Create model
nb = MultinomialNB(alpha=0.1)
# Train model
nb.fit(X_train, y_train)
# Predict
y_pred_nb = nb.predict(X_test)
# Evaluate
print("\nNaive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=["Fake", "Real"]))

# ===============================
# MODEL 3 — RANDOM FOREST
# ===============================
from sklearn.ensemble import RandomForestClassifier

# Create model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# Train model
rf.fit(X_train, y_train)
# Predict
y_pred_rf = rf.predict(X_test)
# Evaluate
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["Fake", "Real"]))

# ===============================
# MODEL 4 — ANN (MLP)
# ===============================
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler

# Scale for ANN (TF-IDF is already sparse, use MaxAbsScaler)
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Create model
ann = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
# Train model
ann.fit(X_train_scaled, y_train)
# Predict
y_pred_ann = ann.predict(X_test_scaled)
# Evaluate
print("\nANN (MLP Classifier):")
print("Accuracy:", accuracy_score(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann, target_names=["Fake", "Real"]))

# ===============================
# CONFUSION MATRICES
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
models_info = [
    ("Logistic Regression", y_pred_lr),
    ("Naive Bayes",         y_pred_nb),
    ("Random Forest",       y_pred_rf),
    ("ANN",                 y_pred_ann),
]
for ax, (name, y_pred) in zip(axes.flat, models_info):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greens",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    ax.set_title(f"{name}\nAcc={accuracy_score(y_test, y_pred)*100:.2f}%")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# ===============================
# FEATURE IMPORTANCE
# ===============================
# Random Forest Feature Importance
feat_names = tfidf.get_feature_names_out()
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.barh([feat_names[i] for i in top_idx[::-1]],
         importances[top_idx[::-1]], color="#378ADD", alpha=0.85)
plt.title("Top 20 Features — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Logistic Regression — Top coefficients (most fake vs most real words)
coef = lr.coef_[0]
top_fake_idx = np.argsort(coef)[:15]         # Most negative = Fake
top_real_idx = np.argsort(coef)[::-1][:15]   # Most positive = Real

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh([feat_names[i] for i in top_fake_idx[::-1]],
             [coef[i] for i in top_fake_idx[::-1]], color="#E24B4A")
axes[0].set_title("Top Words → FAKE (Logistic Regression)")
axes[0].set_xlabel("Coefficient")

axes[1].barh([feat_names[i] for i in top_real_idx[::-1]],
             [coef[i] for i in top_real_idx[::-1]], color="#639922")
axes[1].set_title("Top Words → REAL (Logistic Regression)")
axes[1].set_xlabel("Coefficient")
plt.tight_layout()
plt.show()

# ===============================
# MODEL COMPARISON GRAPH
# ===============================
clf_models = ["Logistic Regression", "Naive Bayes", "Random Forest", "ANN"]
clf_scores = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_ann),
]

plt.figure(figsize=(10, 5))
bars = plt.bar(clf_models, [s * 100 for s in clf_scores],
               color=["#378ADD", "#FF8C00", "#639922", "#9B59B6"])
for bar, score in zip(bars, clf_scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.1,
             f"{score*100:.2f}%", ha="center", va="bottom", fontsize=11)
plt.title("Classification Model Comparison (Accuracy)")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 102)
plt.tight_layout()
plt.show()

# ===============================
# PREDICT WITH NEW INPUT
# ===============================
def predict_news(text, model="rf"):
    clean = preprocess(text)
    vec = tfidf.transform([clean])
    if model == "ann":
        vec = scaler.transform(vec)
        pred = ann.predict(vec)[0]
        prob = ann.predict_proba(vec)[0]
    else:
        pred = rf.predict(vec)[0]
        prob = rf.predict_proba(vec)[0]
    label = "Real" if pred == 1 else "Fake"
    confidence = round(float(prob.max()) * 100, 1)
    return label, confidence

# Demo predictions
demo = [
    "President signs new climate bill after bipartisan Senate vote",
    "SHOCKING deep state globalists plotting secret agenda to destroy America!!!",
    "Federal Reserve raises interest rates citing persistent inflation",
    "You won't believe what they are hiding from you SHARE before deleted",
]

print("\n--- Predictions ---")
for text in demo:
    label, conf = predict_news(text)
    print(f"Text    : {text[:65]}...")
    print(f"Result  : {label}  ({conf}% confidence)\n")
