# ============================================================
# streamlit_app.py — Fraud Detection Dashboard
# ============================================================
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Makes the dashboard look professional
# ============================================================
st.markdown("""
<style>
    .fraud-alert {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .legitimate-alert {
        background-color: #00C851;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px 0;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
        font-size: 20px;
    }
    .risk-medium {
        color: #ffbb33;
        font-weight: bold;
        font-size: 20px;
    }
    .risk-low {
        color: #00C851;
        font-weight: bold;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
# 🧠 We use an environment variable for the API URL so we can
# easily switch between local development and production
# without changing the code. This is a 12-factor app principle.
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def make_prediction(transaction_data: dict):
    """Send transaction to FastAPI and get prediction."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure FastAPI is running."
    except Exception as e:
        return None, str(e)


def get_sample_transactions():
    """
    Return sample transactions for quick testing.
    These are real rows from the dataset — one legitimate, one fraud.
    """
    legitimate = {
        "V1": -1.3598071336738, "V2": -0.0727811733098497,
        "V3": 2.53634673796914, "V4": 1.37815522427443,
        "V5": -0.338320769942518, "V6": 0.462387777762292,
        "V7": 0.239598554061257, "V8": 0.0986979012610507,
        "V9": 0.363786969611213, "V10": 0.0907941719789316,
        "V11": -0.551599533260813, "V12": -0.617800855762348,
        "V13": -0.991389847235408, "V14": -0.311169353699879,
        "V15": 1.46817697209427, "V16": -0.470400525259478,
        "V17": 0.207971241929242, "V18": 0.0257905801985591,
        "V19": 0.403992960255733, "V20": 0.251412098239705,
        "V21": -0.018306777944153, "V22": 0.277837575558899,
        "V23": -0.110473910188767, "V24": 0.0669280749146731,
        "V25": 0.128539358273528, "V26": -0.189114843888824,
        "V27": 0.133558376740387, "V28": -0.0210530534538215,
        "Amount": 149.62, "Time": 0.0
    }
    fraudulent = {
        "V1": -2.3122265423263, "V2": 1.95199201064158,
        "V3": -1.60985073229769, "V4": 3.9979055875468,
        "V5": -0.522187864667764, "V6": -1.42654531920595,
        "V7": -2.53738730624579, "V8": 1.39165724829804,
        "V9": -2.77008927719433, "V10": -2.77227214465915,
        "V11": 3.20203320709635, "V12": -2.89990738849473,
        "V13": -0.595221881324605, "V14": -4.28925378244217,
        "V15": 0.389724120274487, "V16": -1.14074717980657,
        "V17": -2.83005567450437, "V18": -0.0168224681808257,
        "V19": 0.416955705037907, "V20": 0.126910559061474,
        "V21": 0.517232370861764, "V22": -0.0350493686052974,
        "V23": -0.465211076182388, "V24": 0.320198198514526,
        "V25": 0.0445191674731724, "V26": 0.177839798284401,
        "V27": 0.261145002567677, "V28": -0.143275874698919,
        "Amount": 0.0, "Time": 406.0
    }
    return legitimate, fraudulent


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-shield-green.png")
    st.title("🔍 Fraud Detection")
    st.markdown("---")

    # API Status
    st.subheader("🔌 API Status")
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.info("Start FastAPI with:\n`uvicorn app.main:app --reload`")

    st.markdown("---")

    # Model Performance
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC-AUC", "95.18%")
        st.metric("Precision", "91.86%")
    with col2:
        st.metric("F1 Score", "85.87%")
        st.metric("Recall", "80.61%")

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **Dataset:** 284,807 transactions  
    **Model:** Random Forest (100 trees)  
    **Training:** 227,845 transactions  
    **Testing:** 56,962 transactions  
    """)
    st.markdown("---")
    st.markdown("Built by **Amani Azmin**")
    st.markdown("[GitHub](https://github.com/amaniazmin) | [API Docs](http://127.0.0.1:8000/docs)")


# ============================================================
# MAIN PAGE
# ============================================================
st.title("💳 Credit Card Fraud Detection System")
st.markdown("*Production-grade ML system trained on 284,807 real bank transactions*")
st.markdown("---")

# ============================================================
# QUICK TEST SECTION
# ============================================================
st.subheader("⚡ Quick Test with Sample Transactions")
st.markdown("Don't have transaction data? Use these real examples from the dataset:")

col1, col2 = st.columns(2)
legitimate_sample, fraudulent_sample = get_sample_transactions()

with col1:
    if st.button("🟢 Load Legitimate Transaction", use_container_width=True):
        st.session_state.transaction = legitimate_sample
        st.session_state.sample_loaded = "legitimate"

with col2:
    if st.button("🔴 Load Fraud Transaction", use_container_width=True):
        st.session_state.transaction = fraudulent_sample
        st.session_state.sample_loaded = "fraud"

st.markdown("---")

# ============================================================
# TRANSACTION INPUT FORM
# ============================================================
st.subheader("📝 Transaction Details")

# Initialize session state
if "transaction" not in st.session_state:
    st.session_state.transaction = legitimate_sample

txn = st.session_state.transaction

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input(
        "💰 Transaction Amount (€)",
        min_value=0.0,
        max_value=100000.0,
        value=float(txn.get("Amount", 149.62)),
        step=0.01,
        help="The transaction amount in euros"
    )
with col2:
    time_val = st.number_input(
        "⏱️ Time (seconds since first transaction)",
        min_value=0.0,
        max_value=200000.0,
        value=float(txn.get("Time", 0.0)),
        step=1.0,
        help="Seconds elapsed since the first transaction in the dataset"
    )

# V1-V28 features in expandable section
with st.expander("🔬 Advanced: PCA Features (V1-V28) — Click to expand"):
    st.info("These are anonymized PCA-transformed features from the bank. In production these come automatically from the transaction system.")
    cols = st.columns(4)
    v_values = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            v_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=float(st.session_state.transaction.get(f"V{i}", 0.0)),
                format="%.6f",
                key=f"v{i}_{st.session_state.get('sample_loaded', 'default')}"
            )

# ============================================================
# PREDICTION BUTTON
# ============================================================
st.markdown("---")
predict_clicked = st.button(
    "🔍 ANALYZE TRANSACTION",
    use_container_width=True,
    type="primary"
)

if predict_clicked:
    if not api_healthy:
        st.error("❌ Cannot connect to API. Please start FastAPI first.")
    else:
        transaction_data = {
            "Amount": amount,
            "Time": time_val,
            **v_values
        }

        with st.spinner("🔄 Analyzing transaction..."):
            time.sleep(0.5)
            result, error = make_prediction(transaction_data)

        if error:
            st.error(f"❌ Error: {error}")
        else:
            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            if result["prediction"] == 1:
                st.markdown(
                    f'<div class="fraud-alert">🚨 FRAUD DETECTED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="legitimate-alert">✅ LEGITIMATE TRANSACTION</div>',
                    unsafe_allow_html=True
                )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Prediction",
                    result["prediction_label"],
                    delta="⚠️ Fraud" if result["prediction"] == 1 else "✅ Safe"
                )
            with col2:
                st.metric(
                    "Fraud Probability",
                    f"{result['fraud_probability']*100:.2f}%"
                )
            with col3:
                st.metric(
                    "Risk Score",
                    f"{result['risk_score']}/100"
                )
            with col4:
                st.metric(
                    "Confidence",
                    result["confidence"]
                )

            risk = result["risk_score"]
            if risk >= 70:
                risk_class = "risk-high"
            elif risk >= 30:
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"

            st.markdown("**📋 Analysis Message:**")
            st.info(result["message"])

            st.markdown("**📊 Risk Level:**")
            st.progress(result["risk_score"] / 100)

            with st.expander("🔧 Raw API Response (JSON)"):
                st.json(result)
                