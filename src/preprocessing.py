import pandas as pd
import re
import os

# ─────────────────────────────────────────────
# NLTK Setup (unconditional)
# ─────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────────
# CATEGORY MAPPING
# 3 Main Categories + Subcategories (intents)
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    # ── ACCOUNTS ──────────────────────────────
    'create_account':           'Accounts',
    'delete_account':           'Accounts',
    'edit_account':             'Accounts',
    'recover_password':         'Accounts',
    'registration_problems':    'Accounts',
    'switch_account':           'Accounts',
    # General / Contact → folded into Accounts
    'contact_customer_service': 'Accounts',
    'contact_human_agent':      'Accounts',
    'complaint':                'Accounts',
    'review':                   'Accounts',
    'newsletter_subscription':  'Accounts',

    # ── BILLING ───────────────────────────────
    'payment_issue':            'Billing',
    'check_payment_methods':    'Billing',
    'check_invoice':            'Billing',
    'get_invoice':              'Billing',
    'check_refund_policy':      'Billing',
    'get_refund':               'Billing',
    'track_refund':             'Billing',
    'check_cancellation_fee':   'Billing',

    # ── ORDERS ────────────────────────────────
    'cancel_order':             'Orders',
    'change_order':             'Orders',
    'place_order':              'Orders',
    'track_order':              'Orders',
    'change_shipping_address':  'Orders',
    'set_up_shipping_address':  'Orders',
    'delivery_options':         'Orders',
    'delivery_period':          'Orders',
}

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — KEYWORD FLAG COLUMNS
# These numeric columns are trained ON by the ML model
# so the model itself learns keyword → priority patterns.
# This is pure Feature Engineering (not post-prediction rules).
# ─────────────────────────────────────────────

# Critical keywords: severe financial / security issues
# NOTE: "deduct" never appears in training data, but we
# engineer this feature so ANY future ticket with this word
# is correctly flagged before the model even predicts.
CRITICAL_KEYWORDS = [
    'deduct', 'charge', 'overcharge', 'double charged',
    'debit', 'unauthorized', 'fraud', 'scam',
    'stolen', 'money missing', 'money gone',
    'wrong amount', 'incorrect charge',
]

# High urgency keywords (base lemmas)
HIGH_KEYWORDS = [
    'urgent', 'immediate', 'asap', 'emergency',
    'lock out', 'cant access', "cant access",
    'cannot access', 'account block', 'suspend',
    'break', 'not work', 'fail', 'error',
    'cancel refund', 'refund cancel',
]

# Low priority keywords (base lemmas)
LOW_KEYWORDS = [
    'newsletter', 'subscription', 'curious',
    'information', 'general question', 'wonder',
    'no rush', 'whenever', 'feedback', 'suggestion',
    'review', 'survey',
]

def has_critical_keywords(text):
    """Feature: returns 1 if text contains any critical keyword, else 0."""
    text = str(text).lower()
    return int(any(kw in text for kw in CRITICAL_KEYWORDS))


def has_high_keywords(text):
    """Feature: returns 1 if text contains any high-urgency keyword, else 0."""
    text = str(text).lower()
    return int(any(kw in text for kw in HIGH_KEYWORDS))


def has_low_keywords(text):
    """Feature: returns 1 if text contains any low-priority keyword, else 0."""
    text = str(text).lower()
    return int(any(kw in text for kw in LOW_KEYWORDS))


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text):
    """Removes noise, stopwords, and lemmatizes the text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)


# ─────────────────────────────────────────────
# PRIORITY ASSIGNMENT (Feature Engineering)
# Uses BOTH intent label + keyword flags
# This creates the target column the priority ML model trains on.
# ─────────────────────────────────────────────

# Intent → base priority (learned from domain knowledge)
INTENT_PRIORITY_MAP = {
    # Critical
    'payment_issue':            'Critical',

    # High
    'cancel_order':             'High',
    'recover_password':         'High',
    'registration_problems':    'High',
    'get_refund':               'High',
    'check_cancellation_fee':   'High',
    'contact_human_agent':      'High',
    'complaint':                'High',

    # Medium
    'track_refund':             'Medium',
    'track_order':              'Medium',
    'change_order':             'Medium',
    'check_invoice':            'Medium',
    'get_invoice':              'Medium',
    'check_payment_methods':    'Medium',
    'change_shipping_address':  'Medium',
    'delete_account':           'Medium',
    'edit_account':             'Medium',
    'switch_account':           'Medium',

    # Low
    'delivery_options':         'Low',
    'delivery_period':          'Low',
    'check_refund_policy':      'Low',
    'newsletter_subscription':  'Low',
    'create_account':           'Low',
    'place_order':              'Low',
    'set_up_shipping_address':  'Low',
    'contact_customer_service': 'Low',
    'review':                   'Low',
}


def assign_priority(row):
    """
    Assigns priority using Feature Engineering:
    1. Keyword flags (engineered features) take precedence
    2. Intent-based mapping fills the rest
    3. Safety net: Medium
    """
    critical_flag = row['has_critical_keyword']
    high_flag     = row['has_high_keyword']
    low_flag      = row['has_low_keyword']
    intent        = str(row['specific_intent']).lower()

    # Keyword-based override (Feature Engineering layer)
    if critical_flag == 1:
        return 'Critical'
    if high_flag == 1:
        return 'High'

    # Intent-based assignment
    base = INTENT_PRIORITY_MAP.get(intent, None)
    if base:
        # Low keyword can downgrade Medium → Low
        if low_flag == 1 and base == 'Medium':
            return 'Low'
        return base

    # Safety net
    return 'Medium'


# ─────────────────────────────────────────────
# PHASE 1 PIPELINE
# ─────────────────────────────────────────────
def run_phase_1(input_path, output_path):
    print("─" * 50)
    print("  PHASE 1: Data Processing & Feature Engineering")
    print("─" * 50)

    # 1. Load
    df = pd.read_csv(input_path)
    print(f"✔ Loaded {len(df)} rows | Columns: {df.columns.tolist()}")

    # 2. Rename columns
    df = df.rename(columns={'utterance': 'raw_text', 'intent': 'specific_intent'})

    # Remove dead weight columns early (safe if absent)
    df = df.drop(columns=['tags', 'category'], errors='ignore')

    # 3. Map to 3 main categories
    print("✔ Mapping intents → 3 main categories (Accounts / Billing / Orders)...")
    df['main_category'] = df['specific_intent'].map(CATEGORY_MAP)
    unmapped = df['main_category'].isna().sum()
    if unmapped > 0:
        print(f"  ⚠ Warning: {unmapped} rows had unmapped intents → defaulting to 'Accounts'")
        df['main_category'] = df['main_category'].fillna('Accounts')

    # 4. Clean text
    print("✔ Cleaning ticket text (stopwords, lemmatization)...")
    df['cleaned_text'] = df['raw_text'].apply(clean_text)

    # 5. Engineer keyword flag features
    print("✔ Engineering keyword flag features...")
    df['has_critical_keyword'] = df['cleaned_text'].apply(has_critical_keywords)
    df['has_high_keyword']     = df['cleaned_text'].apply(has_high_keywords)
    df['has_low_keyword']      = df['cleaned_text'].apply(has_low_keywords)
    print(f"  Critical flags: {df['has_critical_keyword'].sum()}")
    print(f"  High flags:     {df['has_high_keyword'].sum()}")
    print(f"  Low flags:      {df['has_low_keyword'].sum()}")

    # 6. Assign priority using feature engineering
    print("✔ Assigning priority labels using Feature Engineering...")
    df['priority'] = df.apply(assign_priority, axis=1)
    print(f"  Priority distribution:\n{df['priority'].value_counts().to_string()}")

    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Phase 1 Complete! Saved {len(df)} processed rows → {output_path}")
    print("─" * 50)

    return df


if __name__ == "__main__":
    INPUT  = "data/raw/Bitext_Sample_Customer_Service_Training_Dataset.csv"
    OUTPUT = "data/processed/cleaned_tickets.csv"
    run_phase_1(INPUT, OUTPUT)
