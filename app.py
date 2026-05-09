import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# ===============================
# PAGE SETTINGS
# ===============================
st.set_page_config(layout="wide", page_title="Fake News Detector")

# ===============================
# STOPWORDS
# ===============================
STOP = set(["the","a","an","and","or","but","in","on","at","to","for","of",
            "with","by","from","is","are","was","were","be","been","have",
            "has","had","do","does","did","will","would","could","should",
            "it","its","this","that","i","we","you","he","she","they",
            "me","us","him","her","them","my","our","your","his","their",
            "what","which","who","as","also","not","no","all","so","just","said"])

# ===============================
# PREPROCESSING
# ===============================
def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(words)

# ===============================
# LOAD & TRAIN (cached)
# ===============================
@st.cache_resource
def load_and_train():
    fake = pd.read_csv("Dataset/fake.csv")
    true = pd.read_csv("Dataset/true.csv")
    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    data["clean_text"] = (data["title"].fillna("") + " " + data["text"].fillna("")).apply(preprocess)
    data["text_len"]   = data["text"].fillna("").apply(lambda x: len(x.split()))

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = tfidf.fit_transform(data["clean_text"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models
    lr  = LogisticRegression(max_iter=1000, C=1.0)
    nb  = MultinomialNB(alpha=0.1)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    lr.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    ann = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    ann.fit(X_train_s, y_train)

    accuracies = {
        "Logistic Regression": accuracy_score(y_test, lr.predict(X_test)),
        "Naive Bayes":         accuracy_score(y_test, nb.predict(X_test)),
        "Random Forest":       accuracy_score(y_test, rf.predict(X_test)),
        "ANN":                 accuracy_score(y_test, ann.predict(X_test_s)),
    }

    return data, tfidf, scaler, lr, nb, rf, ann, X_train, X_test, y_train, y_test, accuracies

# ===============================
# LOAD
# ===============================
with st.spinner("Training models, please wait..."):
    data, tfidf, scaler, lr, nb, rf, ann, X_train, X_test, y_train, y_test, accuracies = load_and_train()

fake_data = data[data["label"] == 0]
real_data = data[data["label"] == 1]

# ===============================
# UI — TITLE
# ===============================
st.title("📰 Fake News Detection System")
st.markdown("Detects whether a news article is **Real** or **Fake** using 4 ML models.")

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model Performance", "📈 EDA"])

# ──────────────────────────────────────────
# TAB 1 — PREDICT
# ──────────────────────────────────────────
with tab1:
    st.header("Enter News Article")

    col1, col2 = st.columns(2)

    with col1:
        title_input = st.text_input("News Headline / Title")
        model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "Naive Bayes", "ANN"])

    with col2:
        text_input = st.text_area("News Body (optional — paste article text)", height=150)

    if st.button("🔍 Predict"):

        full_text = title_input + " " + text_input
        clean = preprocess(full_text)
        vec = tfidf.transform([clean])

        if model_choice == "ANN":
            vec_s = scaler.transform(vec)
            pred  = ann.predict(vec_s)[0]
            probs = ann.predict_proba(vec_s)[0]
        elif model_choice == "Logistic Regression":
            pred  = lr.predict(vec)[0]
            probs = lr.predict_proba(vec)[0]
        elif model_choice == "Naive Bayes":
            pred  = nb.predict(vec)[0]
            probs = nb.predict_proba(vec)[0]
        else:
            pred  = rf.predict(vec)[0]
            probs = rf.predict_proba(vec)[0]

        label      = "Real" if pred == 1 else "Fake"
        fake_prob  = round(probs[0] * 100, 2)
        real_prob  = round(probs[1] * 100, 2)
        confidence = max(fake_prob, real_prob)

        # ── Results ──
        st.subheader("🧾 Results")
        c1, c2 = st.columns(2)
        with c1:
            color = "#4CAF50" if label == "Real" else "#F44336"
            st.markdown(f"<h2 style='color:{color};'>Prediction</h2><h1>{label}</h1>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<h2 style='color:#2196F3;'>Confidence</h2><h1>{confidence}%</h1>", unsafe_allow_html=True)

        # ── Confidence bar ──
        st.subheader("📊 Prediction Confidence")
        st.write(f"🔴 Fake → {fake_prob}%")
        st.progress(fake_prob / 100)
        st.write(f"🟢 Real → {real_prob}%")
        st.progress(real_prob / 100)

        # ── Risk Level ──
        st.subheader("⚠️ Risk Level")
        if label == "Fake" and confidence > 90:
            st.error("🔴 High Risk — Very likely fake news")
        elif label == "Fake":
            st.warning("🟠 Moderate Risk — Possibly fake news")
        else:
            st.success("🟢 Low Risk — Likely real news")

        # ── Why this result ──
        st.subheader("🧠 Why this result")
        reasons = []
        text_lower = full_text.lower()

        fake_signals = ["deep state", "globalist", "mainstream media", "you won't believe",
                        "share before", "they don't want you", "secret", "exposed", "hoax",
                        "cover-up", "witch hunt", "rigged", "corrupt", "shocking"]
        real_signals = ["reuters", "according to", "officials said", "confirmed", "reported",
                        "percent", "billion", "senate", "congress", "department", "statement",
                        "press conference", "spokesperson"]

        for sig in fake_signals:
            if sig in text_lower:
                reasons.append(f"Contains sensationalist phrase: **'{sig}'** — common in fake news")
        for sig in real_signals:
            if sig in text_lower:
                reasons.append(f"Contains credible journalism signal: **'{sig}'** — common in real news")

        word_count = len(full_text.split())
        if word_count < 20:
            reasons.append("Very short text — model has limited information to analyse")
        if full_text.isupper() and len(full_text) > 10:
            reasons.append("All-caps writing style is common in sensationalist fake news")
        if full_text.count("!") > 2:
            reasons.append("Excessive exclamation marks are a fake news signal")

        if reasons:
            for r in reasons:
                st.markdown(f"• {r}")
        else:
            st.write("No specific strong signals detected — prediction based on overall word patterns.")

        # ── Suggestions ──
        st.subheader("💡 Suggestions")
        suggestions = []
        if label == "Fake":
            suggestions = [
                "Cross-check this article with trusted sources (BBC, Reuters, AP News)",
                "Look for the original source — is there a byline or author name?",
                "Search for the same story on multiple reliable news sites",
                "Check if the URL is from a known news organisation",
                "Use fact-checking sites like Snopes, FactCheck.org, or PolitiFact",
            ]
        else:
            suggestions = [
                "Article appears credible — still good practice to verify key claims",
                "Check the publication date — old articles are sometimes shared as new",
                "Ensure you are reading the full article, not just the headline",
            ]
        for s in suggestions:
            st.write("•", s)

        # ── All models comparison ──
        st.subheader("🤖 All Models on this Article")
        clean_vec = tfidf.transform([clean])
        results = {}
        results["Logistic Regression"] = lr.predict_proba(clean_vec)[0]
        results["Naive Bayes"]         = nb.predict_proba(clean_vec)[0]
        results["Random Forest"]       = rf.predict_proba(clean_vec)[0]
        results["ANN"]                 = ann.predict_proba(scaler.transform(clean_vec))[0]

        for mname, prb in results.items():
            mpred = "Real ✅" if prb[1] > 0.5 else "Fake ❌"
            mconf = round(max(prb) * 100, 1)
            st.write(f"**{mname}** → {mpred}  ({mconf}% confidence)")

# ──────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────
with tab2:
    st.header("Model Performance Comparison")

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#378ADD", "#FF8C00", "#639922", "#9B59B6"]
    bars = ax.bar(accuracies.keys(), [v * 100 for v in accuracies.values()], color=colors)
    for bar, (name, score) in zip(bars, accuracies.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{score*100:.2f}%", ha="center", va="bottom", fontsize=11)
    ax.set_title("Classification Model Comparison (Accuracy)")
    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(90, 102)
    plt.tight_layout()
    st.pyplot(fig)

    # Metrics table
    st.subheader("📋 Accuracy Table")
    acc_df = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "Accuracy (%)": [round(v * 100, 2) for v in accuracies.values()]
    }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    st.dataframe(acc_df, use_container_width=True)

    # Confusion matrices
    st.subheader("🔲 Confusion Matrices")
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    vec_test = X_test
    vec_test_s = scaler.transform(X_test)
    models_pred = [
        ("Logistic Regression", lr.predict(vec_test)),
        ("Naive Bayes",         nb.predict(vec_test)),
        ("Random Forest",       rf.predict(vec_test)),
        ("ANN",                 ann.predict(vec_test_s)),
    ]
    for ax, (mname, y_pred) in zip(axes.flat, models_pred):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Greens",
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{mname}  |  Acc={acc*100:.2f}%")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    st.pyplot(fig2)

    # Feature importance — Random Forest
    st.subheader("🌲 Top 20 Features — Random Forest")
    feat_names = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh([feat_names[i] for i in top_idx[::-1]],
             importances[top_idx[::-1]], color="#378ADD", alpha=0.85)
    ax3.set_title("Top 20 Most Important Words — Random Forest")
    ax3.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig3)

    # Logistic Regression coefficients
    st.subheader("📐 Top Words → Fake & Real (Logistic Regression)")
    coef = lr.coef_[0]
    top_fake_idx = np.argsort(coef)[:15]
    top_real_idx = np.argsort(coef)[::-1][:15]

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    axes4[0].barh([feat_names[i] for i in top_fake_idx[::-1]],
                  [coef[i] for i in top_fake_idx[::-1]], color="#E24B4A")
    axes4[0].set_title("Top Words → FAKE")
    axes4[0].set_xlabel("Coefficient")

    axes4[1].barh([feat_names[i] for i in top_real_idx[::-1]],
                  [coef[i] for i in top_real_idx[::-1]], color="#639922")
    axes4[1].set_title("Top Words → REAL")
    axes4[1].set_xlabel("Coefficient")
    plt.tight_layout()
    st.pyplot(fig4)

# ──────────────────────────────────────────
# TAB 3 — EDA
# ──────────────────────────────────────────
with tab3:
    st.header("Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", f"{len(data):,}")
    col2.metric("Fake Articles",  f"{(data.label==0).sum():,}")
    col3.metric("Real Articles",  f"{(data.label==1).sum():,}")
    col4.metric("Best Accuracy",  f"{max(accuracies.values())*100:.2f}%")

    # Class balance
    st.subheader("Class Distribution")
    fig_a, axes_a = plt.subplots(1, 2, figsize=(12, 4))
    data["label"].map({0: "Fake", 1: "Real"}).value_counts().plot(
        kind="bar", ax=axes_a[0], color=["#E24B4A", "#639922"])
    axes_a[0].set_title("Fake vs Real Count")
    axes_a[0].set_xlabel("Type")
    axes_a[0].set_ylabel("Count")
    axes_a[0].tick_params(axis="x", rotation=0)

    axes_a[1].pie([len(fake_data), len(real_data)],
                  labels=["Fake", "Real"], colors=["#E24B4A", "#639922"],
                  autopct="%1.1f%%", startangle=90)
    axes_a[1].set_title("Class Balance")
    plt.tight_layout()
    st.pyplot(fig_a)

    # Article length
    st.subheader("Article Length Distribution")
    fig_b, ax_b = plt.subplots(figsize=(10, 4))
    ax_b.hist(data[data.label == 0]["text_len"].clip(upper=1500), bins=50,
              alpha=0.6, color="#E24B4A", label="Fake")
    ax_b.hist(data[data.label == 1]["text_len"].clip(upper=1500), bins=50,
              alpha=0.6, color="#639922", label="Real")
    ax_b.set_title("Article Length Distribution (Words)")
    ax_b.set_xlabel("Word Count")
    ax_b.set_ylabel("Frequency")
    ax_b.legend()
    plt.tight_layout()
    st.pyplot(fig_b)

    # You vs Average stats
    st.subheader("📊 Dataset Statistics")
    stats_df = pd.DataFrame({
        "Metric":      ["Avg Article Length", "Median Article Length"],
        "Fake News":   [
            round(data[data.label == 0]["text_len"].mean()),
            round(data[data.label == 0]["text_len"].median()),
        ],
        "Real News":   [
            round(data[data.label == 1]["text_len"].mean()),
            round(data[data.label == 1]["text_len"].median()),
        ],
    })
    st.dataframe(stats_df, use_container_width=True)

    # Top words
    st.subheader("Top Words — Fake vs Real")

    def top_words(series, n=15):
        words = []
        for text in series.fillna(""):
            words.extend([w for w in re.sub(r"[^a-z\s]","",text.lower()).split()
                          if w not in STOP and len(w) > 3])
        return Counter(words).most_common(n)

    fake_top = top_words(fake_data["text"])
    true_top = top_words(real_data["text"])

    fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 6))
    wf, ff = zip(*fake_top)
    axes_c[0].barh(wf[::-1], ff[::-1], color="#E24B4A", alpha=0.85)
    axes_c[0].set_title("Top Words — Fake News")
    axes_c[0].set_xlabel("Frequency")

    wt, ft = zip(*true_top)
    axes_c[1].barh(wt[::-1], ft[::-1], color="#639922", alpha=0.85)
    axes_c[1].set_title("Top Words — Real News")
    axes_c[1].set_xlabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig_c)

    # Subject distributions
    st.subheader("Subject Distribution")
    fig_d, axes_d = plt.subplots(1, 2, figsize=(14, 4))
    fake_data["subject"].value_counts().plot(kind="bar", ax=axes_d[0], color="#E24B4A")
    axes_d[0].set_title("Fake News — Subjects")
    axes_d[0].tick_params(axis="x", rotation=30)

    real_data["subject"].value_counts().plot(kind="bar", ax=axes_d[1], color="#639922")
    axes_d[1].set_title("Real News — Subjects")
    axes_d[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig_d)
