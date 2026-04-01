import streamlit as st
import joblib
import pandas as pd
import sys
import os
import zipfile
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import clean_text
from src.predict_bulk import predict_single, predict_bulk

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TicketTriage | AI Support",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

    .metric-card {
        background: white;
        border: 1px solid #e8edf2;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; margin-top: 0.3rem; }

    .result-box {
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2d6a9f;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .priority-Critical { background:#fef2f2; border-color:#fca5a5; border-left-color:#dc2626; }
    .priority-High     { background:#fff7ed; border-color:#fdba74; border-left-color:#ea580c; }
    .priority-Medium   { background:#fefce8; border-color:#fde047; border-left-color:#ca8a04; }
    .priority-Low      { background:#f0fdf4; border-color:#86efac; border-left-color:#16a34a; }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-Critical { background:#fee2e2; color:#dc2626; }
    .badge-High     { background:#ffedd5; color:#ea580c; }
    .badge-Medium   { background:#fef9c3; color:#a16207; }
    .badge-Low      { background:#dcfce7; color:#16a34a; }
    .badge-Accounts { background:#ede9fe; color:#7c3aed; }
    .badge-Billing  { background:#dbeafe; color:#1d4ed8; }
    .badge-Orders   { background:#d1fae5; color:#065f46; }

    .rule-alert {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 0.8rem;
    }

    .sidebar-info {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #374151;
        border: 1px solid #e2e8f0;
    }

    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# helper for clear button
def clear_text():
    st.session_state.user_input = ""

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        cat_model  = joblib.load("models/category_model.pkl")
        prio_model = joblib.load("models/priority_model.pkl")
        scores     = joblib.load("models/model_scores.pkl")
        return vectorizer, cat_model, prio_model, scores
    except Exception as e:
        return None, None, None, None

vectorizer, cat_model, prio_model, scores = load_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎫 TicketTriage")
    st.markdown("---")

    if scores:
        st.markdown("### 🧠 Model Performance")
        st.metric("Category Accuracy", f"{scores['category_accuracy']}%")
        st.metric("Priority Accuracy", f"{scores['priority_accuracy']}%")
        st.markdown("---")

    st.caption("TicketTriage v1.0 | NLP + SVM + Feature Engineering")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🎫 TicketTriage</h1>
    <p>AI-Powered Customer Support Ticket Classification & Priority Detection</p>
</div>
""", unsafe_allow_html=True)

if vectorizer is None:
    st.error("⚠️ Models not found! Please run `python src/train_model.py` first.")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "💬 Single Ticket Analysis",
    "📁 Bulk CSV Upload",
    "📊 Session Dashboard"
])

# ══════════════════════════════════════════════
# TAB 1: SINGLE TICKET
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Submit a Customer Ticket")
        user_input = st.text_area(
            "Enter Customer Message:",
            value=st.session_state.user_input,
            key="user_input",
            height=150
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            analyze_btn = st.button("🔮 Analyze Ticket", type="primary", use_container_width=True)
        with col_btn2:
            st.button("Clear", on_click=clear_text, use_container_width=True)

        if analyze_btn:
            trimmed_input = st.session_state.user_input.strip()
            if len(trimmed_input) < 10 or " " not in trimmed_input:
                st.warning("Please enter a valid customer message (at least 10 characters).")
            else:
                with st.spinner("AI is analyzing your ticket..."):
                    result = predict_single(trimmed_input, vectorizer, cat_model, prio_model)

                # Add to history
                st.session_state.history.append({
                    "Ticket (preview)":  trimmed_input[:60] + "..." if len(trimmed_input) > 60 else trimmed_input,
                    "Category":          result['category'],
                    "Subcategory":       result['subcategory'],
                    "AI Priority":       result['ai_priority'],
                    "Final Priority":    result['final_priority'],
                    "Rule Applied":      "✅ Yes" if result['rule_applied'] else "—",
                })

                # Display results
                st.success("✅ Analysis Complete!")

                prio = result['final_priority']
                cat  = result['category']

                st.markdown(f"""
                <div class='result-box priority-{prio}'>
                    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;'>
                        <div>
                            <div style='font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase;'>CATEGORY</div>
                            <div style='font-size:1.4rem; font-weight:700; color:#1e3a5f;'>
                                <span class='badge badge-{cat}'>{cat}</span>
                            </div>
                            <div style='font-size:0.85rem; color:#6b7280; margin-top:0.3rem;'>
                                Subcategory: {result['subcategory']}
                            </div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase;'>PRIORITY</div>
                            <span class='badge badge-{prio}' style='font-size:1.1rem; padding:0.5rem 1.2rem;'>{prio}</span>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:0.75rem; color:#6b7280; font-weight:600; text-transform:uppercase;'>AI BASE PRIORITY</div>
                            <div style='font-size:1rem; font-weight:600; color:#374151;'>{result['ai_priority']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if result['rule_applied']:
                    st.markdown(f"""
                    <div class='rule-alert'>
                        ⚡ <b>Business Rule Triggered:</b> {result['rule_reason']}<br>
                        Priority upgraded from <b>{result['ai_priority']}</b> → <b>{result['final_priority']}</b>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📋 Quick Reference")
        st.markdown("""
        <div class='sidebar-info'>
        <b>👤 Accounts</b><br>
        Login, Registration, Password, Profile, Complaints, Contact<br><br>
        <b>💳 Billing</b><br>
        Payments, Invoices, Refunds, Cancellation Fees<br><br>
        <b>📦 Orders</b><br>
        Track, Cancel, Change Orders, Delivery, Shipping
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div class='sidebar-info'>
        <span class='badge badge-Critical'>🔴 Critical</span> Fraud, unauthorized charges<br><br>
        <span class='badge badge-High'>🟠 High</span> Urgent, locked out, failed<br><br>
        <span class='badge badge-Medium'>🟡 Medium</span> Tracking, invoices, changes<br><br>
        <span class='badge badge-Low'>🟢 Low</span> General info, newsletters
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2: BULK CSV UPLOAD
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Bulk Ticket Processing")
    st.markdown("Upload your CSV file. The system will classify every ticket and detect priority automatically.")

    uploaded_file = st.file_uploader(
        "Upload CSV (must have an 'utterance' or 'instruction' column)",
        type=["csv"]
    )

    if uploaded_file:
        bulk_df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded:** `{uploaded_file.name}` — {len(bulk_df):,} rows, {len(bulk_df.columns)} columns")
        st.dataframe(bulk_df.head(5), use_container_width=True, hide_index=True)

        # Detect text column
        text_col = None
        for candidate in ['utterance', 'instruction', 'text', 'message', 'ticket']:
            if candidate in bulk_df.columns:
                text_col = candidate
                break
        if text_col is None:
            text_col = bulk_df.columns[0]

        st.info(f"📌 Using column **`{text_col}`** as the ticket text source.")

        if st.button("⚙️ Process All Tickets", type="primary"):
            with st.spinner(f"Processing {len(bulk_df):,} tickets..."):
                result_df = predict_bulk(
                    bulk_df.copy(), text_col, vectorizer, cat_model, prio_model
                )

            st.success(f"✅ Processed {len(result_df):,} tickets!")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            priority_counts = result_df['Final_Priority'].value_counts()
            rules_applied   = (result_df['Business_Rule_Flag'] != 'False').sum()

            with m1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>🔴 Critical</div>
                    <div class='value'>{priority_counts.get('Critical', 0)}</div></div>""",
                    unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>🟠 High</div>
                    <div class='value'>{priority_counts.get('High', 0)}</div></div>""",
                    unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>🟡 Medium</div>
                    <div class='value'>{priority_counts.get('Medium', 0)}</div></div>""",
                    unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>⚡ Rules Fired</div>
                    <div class='value'>{rules_applied}</div></div>""",
                    unsafe_allow_html=True)

            st.markdown("#### Preview of Results")
            display_cols = [text_col, 'Predicted_Category', 'AI_Priority', 'Final_Priority', 'Business_Rule_Flag']
            display_cols = [c for c in display_cols if c in result_df.columns]
            st.dataframe(result_df[display_cols].head(20), use_container_width=True, hide_index=True)

            # Download routed queues as ZIP of CSVs (mutually exclusive)
            urgent_mask = result_df['Final_Priority'] == 'Critical'
            urgent_df = result_df[urgent_mask]
            standard_df = result_df[~urgent_mask]

            accounts_df = standard_df[standard_df['Predicted_Category'] == 'Accounts']
            billing_df  = standard_df[standard_df['Predicted_Category'] == 'Billing']
            orders_df   = standard_df[standard_df['Predicted_Category'] == 'Orders']

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('Accounts_Queue.csv', accounts_df.to_csv(index=False))
                zf.writestr('Billing_Queue.csv', billing_df.to_csv(index=False))
                zf.writestr('Orders_Queue.csv', orders_df.to_csv(index=False))
                zf.writestr('Urgent_Escalations.csv', urgent_df.to_csv(index=False))

            buffer.seek(0)
            st.download_button(
                label="📥 Download Routed Queues (ZIP)",
                data=buffer.getvalue(),
                file_name='TicketTriage_Routed_Queues.zip',
                mime='application/zip',
                type='primary',
                use_container_width=True
            )


# ══════════════════════════════════════════════
# TAB 3: SESSION DASHBOARD
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📊 Session Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.info("No tickets analyzed yet in this session. Use the Single Ticket tab to get started.")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        # Summary metrics
        total = len(history_df)
        critical_count = (history_df['Final Priority'] == 'Critical').sum()
        high_count     = (history_df['Final Priority'] == 'High').sum()
        rules_count    = (history_df['Rule Applied'] == '✅ Yes').sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Total Tickets</div>
                <div class='value'>{total}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>🔴 Critical</div>
                <div class='value'>{critical_count}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>🟠 High</div>
                <div class='value'>{high_count}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>⚡ Rules Applied</div>
                <div class='value'>{rules_count}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Category Distribution")
            cat_counts = history_df['Category'].value_counts()
            st.bar_chart(cat_counts)

        with col_b:
            st.markdown("#### Priority Distribution")
            prio_counts = history_df['Final Priority'].value_counts()
            st.bar_chart(prio_counts)

        st.markdown("#### Full Session History")
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear Session History"):
            st.session_state.history = []
            st.rerun()
