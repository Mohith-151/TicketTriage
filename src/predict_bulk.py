"""
predict_bulk.py — Business Rule Layer (Post-Prediction Safety Net)

This runs AFTER the ML model predicts priority.
It catches edge cases the training data never contained
(e.g. "deduct", "fraud", "stolen") and overrides to Critical.

This is NOT replacing the AI — it is a safety net for
the ~5% of real-world tickets with words the model never saw.
"""

import sys
from pathlib import Path

# Add workspace root to path so 'src' module can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    CRITICAL_KEYWORDS,
    HIGH_KEYWORDS,
    clean_text,
    has_critical_keywords,
    has_high_keywords,
    has_low_keywords,
)
from scipy.sparse import hstack, csr_matrix
import pandas as pd


# ─────────────────────────────────────────────
# BUSINESS RULE — POST-PREDICTION OVERRIDE
# ─────────────────────────────────────────────
def apply_business_rules(cleaned_text, ai_predicted_priority):
    """
    Applies business rules after AI prediction.
    Returns (final_priority, rule_applied, rule_reason).

    Rule hierarchy:
      1. Critical keyword found → always Critical
      2. High keyword found + AI said Low → upgrade to High
      3. No rule triggered → trust the AI
    """
    text = str(cleaned_text).lower()

    # Rule 1: Critical keywords override everything
    triggered_critical = [kw for kw in CRITICAL_KEYWORDS if kw in text]
    if triggered_critical:
        return 'Critical', True, f"Keyword detected: '{triggered_critical[0]}'"

    # Rule 2: High keyword present but AI underestimated
    triggered_high = [kw for kw in HIGH_KEYWORDS if kw in text]
    if triggered_high and ai_predicted_priority == 'Low':
        return 'High', True, f"Urgency keyword detected: '{triggered_high[0]}'"

    # Rule 3: Trust the AI
    return ai_predicted_priority, False, None


# ─────────────────────────────────────────────
# SINGLE TICKET PREDICTION HELPER
# (used by app.py for both single + bulk)
# ─────────────────────────────────────────────
def predict_single(raw_text, vectorizer, cat_model, prio_model):
    """
    Full prediction pipeline for a single ticket:
    1. Clean text
    2. Build feature matrix (TF-IDF + keyword flags)
    3. AI predicts category + priority
    4. Business rules applied as safety net
    Returns dict with all results.
    """
    # Step 1: Clean
    cleaned = clean_text(raw_text)

    # Step 2: Build feature matrix (must match training structure)
    X_tfidf = vectorizer.transform([cleaned])
    keyword_flags = csr_matrix([[
        has_critical_keywords(cleaned),
        has_high_keywords(cleaned),
        has_low_keywords(cleaned),
    ]])
    X = hstack([X_tfidf, keyword_flags])

    # Step 3: AI predictions
    ai_category = cat_model.predict(X)[0]
    ai_priority = prio_model.predict(X)[0]

    # Step 4: Business rule safety net
    final_priority, rule_applied, rule_reason = apply_business_rules(cleaned, ai_priority)

    # Subcategory = specific intent (approximated from category for display)
    subcategory_map = {
        'Accounts': 'Account Management',
        'Billing':  'Payment & Invoicing',
        'Orders':   'Order & Delivery',
    }

    return {
        'category':       ai_category,
        'subcategory':    subcategory_map.get(ai_category, ai_category),
        'ai_priority':    ai_priority,
        'final_priority': final_priority,
        'rule_applied':   rule_applied,
        'rule_reason':    rule_reason,
    }


# ─────────────────────────────────────────────
# BULK PREDICTION
# ─────────────────────────────────────────────
def predict_bulk(df, text_col, vectorizer, cat_model, prio_model):
    """
    Runs full prediction pipeline on a DataFrame.
    Returns the DataFrame with new prediction columns added.
    """
    # Clean text
    df['cleaned_text'] = df[text_col].astype(str).apply(clean_text)

    # Build feature matrix
    X_tfidf = vectorizer.transform(df['cleaned_text'])
    keyword_flags = csr_matrix(
        df['cleaned_text'].apply(lambda t: [
            has_critical_keywords(t),
            has_high_keywords(t),
            has_low_keywords(t),
        ]).tolist()
    )
    X = hstack([X_tfidf, keyword_flags])

    # AI predictions
    df['Predicted_Category'] = cat_model.predict(X)
    ai_priorities            = prio_model.predict(X)

    # Apply business rules per row
    results = [
        apply_business_rules(text, ai_prio)
        for text, ai_prio in zip(df['cleaned_text'], ai_priorities)
    ]
    df['AI_Priority']        = ai_priorities
    df['Final_Priority']     = [r[0] for r in results]
    df['Business_Rule_Flag'] = [r[2] if r[1] else 'False' for r in results]

    return df
