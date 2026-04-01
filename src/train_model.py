import pandas as pd
import joblib
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import run_phase_1

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
RAW_DATA   = "data/raw/Bitext_Sample_Customer_Service_Training_Dataset.csv"
CLEAN_DATA = "data/processed/cleaned_tickets.csv"
MODEL_DIR  = "models"


def build_feature_matrix(df, vectorizer=None, fit=True):
    """
    Combines TF-IDF vectors with engineered keyword flag columns.
    This is the core of the hybrid approach:
      - TF-IDF learns text patterns
      - Keyword flags teach the model about critical/high/low signals
    """
    # TF-IDF on cleaned text
    if fit:
        X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    else:
        X_tfidf = vectorizer.transform(df['cleaned_text'])

    # Engineered keyword features (sparse matrix)
    keyword_features = csr_matrix(
        df[['has_critical_keyword', 'has_high_keyword', 'has_low_keyword']].values
    )

    # Stack TF-IDF + keyword flags horizontally
    X_combined = hstack([X_tfidf, keyword_features])
    return X_combined


def train():
    print("─" * 50)
    print("  PHASE 2: Model Training")
    print("─" * 50)

    # ── Step 1: Run preprocessing if needed ──
    if not os.path.exists(CLEAN_DATA):
        print("Processed data not found. Running Phase 1 first...")
        run_phase_1(RAW_DATA, CLEAN_DATA)

    df = pd.read_csv(CLEAN_DATA)
    print(f"✔ Loaded {len(df)} processed rows")

    # ── Step 2: Build TF-IDF + keyword feature matrix ──
    print("\n✔ Building feature matrix (TF-IDF + Keyword Flags)...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,
        sublinear_tf=True
    )
    X = build_feature_matrix(df, vectorizer, fit=True)
    print(f"  Feature matrix shape: {X.shape}")

    # ── Step 3: Train CATEGORY Model ──
    print("\n── Training Category Classifier (SVM) ──")
    y_cat = df['main_category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )
    cat_model = LinearSVC(max_iter=2000, C=1.0)
    cat_model.fit(X_train, y_train)
    y_pred_cat = cat_model.predict(X_test)
    cat_acc = accuracy_score(y_test, y_pred_cat)
    print(f"  Category Accuracy: {cat_acc * 100:.2f}%")
    print(classification_report(y_test, y_pred_cat))
    print('\n  Category Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_cat))

    # ── Step 4: Train PRIORITY Model ──
    print("── Training Priority Classifier (SVM) ──")
    y_prio = df['priority']
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X, y_prio, test_size=0.2, random_state=42, stratify=y_prio
    )
    prio_model = LinearSVC(max_iter=2000, C=1.0)
    prio_model.fit(X_train_p, y_train_p)
    y_pred_prio = prio_model.predict(X_test_p)
    prio_acc = accuracy_score(y_test_p, y_pred_prio)
    print(f"  Priority Accuracy: {prio_acc * 100:.2f}%")
    print(classification_report(y_test_p, y_pred_prio))
    print('\n  Priority Confusion Matrix:')
    print(confusion_matrix(y_test_p, y_pred_prio))

    # ── Step 5: Save all models ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer,  f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(cat_model,   f"{MODEL_DIR}/category_model.pkl")
    joblib.dump(prio_model,  f"{MODEL_DIR}/priority_model.pkl")

    # Save accuracy scores for app display
    scores = {
        'category_accuracy': round(cat_acc * 100, 2),
        'priority_accuracy': round(prio_acc * 100, 2),
    }
    joblib.dump(scores, f"{MODEL_DIR}/model_scores.pkl")

    print(f"\n✅ Phase 2 Complete! Models saved to /{MODEL_DIR}/")
    print(f"   Category Accuracy : {cat_acc * 100:.2f}%")
    print(f"   Priority Accuracy : {prio_acc * 100:.2f}%")
    print("─" * 50)


if __name__ == "__main__":
    train()
