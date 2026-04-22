import streamlit as st
import pandas as pd
import joblib
import io
import zipfile
import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict_bulk import predict_bulk

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="TicketTriage | Production", page_icon="🛡️", layout="wide")

if 'ui_step' not in st.session_state:
    st.session_state.ui_step = 'upload'
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_queues' not in st.session_state:
    st.session_state.processed_queues = None

# Polished Branding
st.markdown("""
    <div style='background-color: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 25px;'>
        <h1 style='color: white; margin: 0;'>🛡️ TicketTriage: Enterprise Edition</h1>
        <p style='color: #94a3b8; margin: 5px 0 0 0;'>Automated Customer Support Routing & Queue Separation</p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        cat_model = joblib.load("models/category_model.pkl")
        prio_model = joblib.load("models/priority_model.pkl")
        return vectorizer, cat_model, prio_model
    except:
        return None, None, None

vectorizer, cat_model, prio_model = load_assets()

if vectorizer is None:
    st.error("❌ Model files missing. Please run the training script first.")
    st.stop()

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def _load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Error reading file: {exc}")
        return None


def _to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def _process_queues(df, text_column):
    processed_df = predict_bulk(df.copy(), text_column, vectorizer, cat_model, prio_model)
    ai_cols = ['AI_Priority', 'Final_Priority', 'Business_Rule_Flag', 'cleaned_text']
    urgent_mask = processed_df['Final_Priority'] == 'Critical'
    urgent_out = processed_df[urgent_mask].drop(columns=ai_cols, errors='ignore')
    non_urgent = processed_df[~urgent_mask]
    accounts_out = non_urgent[non_urgent['Predicted_Category'] == 'Accounts'].drop(columns=ai_cols, errors='ignore')
    billing_out = non_urgent[non_urgent['Predicted_Category'] == 'Billing'].drop(columns=ai_cols, errors='ignore')
    orders_out = non_urgent[non_urgent['Predicted_Category'] == 'Orders'].drop(columns=ai_cols, errors='ignore')
    return {
        'urgent': urgent_out,
        'accounts': accounts_out,
        'billing': billing_out,
        'orders': orders_out
    }


def _reset_state():
    st.session_state.ui_step = 'upload'
    st.session_state.raw_data = None
    st.session_state.processed_queues = None


# ─────────────────────────────────────────────
# 1. UPLOAD STEP
# ─────────────────────────────────────────────
if st.session_state.ui_step == 'upload':
    st.subheader("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Customer Tickets (CSV or Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = _load_uploaded_file(uploaded_file)
        if df is not None:
            st.session_state.raw_data = df
            st.session_state.ui_step = 'mapping'
            st.session_state.processed_queues = None
            st.rerun()

# ─────────────────────────────────────────────
# 2. MAPPING STEP
# ─────────────────────────────────────────────
elif st.session_state.ui_step == 'mapping':
    df = st.session_state.raw_data
    if df is None:
        st.warning("No uploaded data found. Please upload a file again.")
        _reset_state()
        st.rerun()

    st.subheader("2. Map Data Columns")
    st.markdown(f"**Uploaded rows:** {len(df)}")
    col_options = df.columns.tolist()
    text_column = st.selectbox(
        "Which column contains the customer's message?",
        options=col_options,
        index=0,
        help="Select the column the AI should analyze for triage."
    )

    if st.button("🚀 Process & Separate Queues", type="primary"):
        with st.spinner("Analyzing tickets and generating queues..."):
            st.session_state.processed_queues = _process_queues(df, text_column)
            st.session_state.ui_step = 'results'
            st.rerun()

# ─────────────────────────────────────────────
# 3. RESULTS STEP
# ─────────────────────────────────────────────
elif st.session_state.ui_step == 'results':
    queues = st.session_state.processed_queues
    if queues is None:
        st.warning("No processed queues available. Please upload and process a file.")
        _reset_state()
        st.rerun()

    urgent_out = queues['urgent']
    accounts_out = queues['accounts']
    billing_out = queues['billing']
    orders_out = queues['orders']

    st.subheader("3. Processing Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Urgent/Critical", len(urgent_out))
    m2.metric("Accounts Queue", len(accounts_out))
    m3.metric("Billing Queue", len(billing_out))
    m4.metric("Orders Queue", len(orders_out))

    st.markdown("---")
    st.subheader("4. Download Routed Queues")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.download_button("🔴 Download Urgent", data=_to_csv_bytes(urgent_out), file_name="Urgent_Escalations.csv", mime="text/csv")
    with d2:
        st.download_button("🟣 Download Accounts", data=_to_csv_bytes(accounts_out), file_name="Accounts_Queue.csv", mime="text/csv")
    with d3:
        st.download_button("🔵 Download Billing", data=_to_csv_bytes(billing_out), file_name="Billing_Queue.csv", mime="text/csv")
    with d4:
        st.download_button("🟢 Download Orders", data=_to_csv_bytes(orders_out), file_name="Orders_Queue.csv", mime="text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('Urgent_Escalations.csv', urgent_out.to_csv(index=False))
        zf.writestr('Accounts_Queue.csv', accounts_out.to_csv(index=False))
        zf.writestr('Billing_Queue.csv', billing_out.to_csv(index=False))
        zf.writestr('Orders_Queue.csv', orders_out.to_csv(index=False))

    st.download_button(
        label="📦 Download All Queues (ZIP)",
        data=buffer.getvalue(),
        file_name="TicketTriage_Routed_Queues.zip",
        mime="application/zip",
        use_container_width=True
    )

    st.markdown("---")
    if st.button("🔄 New File to 'Triage'", use_container_width=True):
        _reset_state()
        st.rerun()
