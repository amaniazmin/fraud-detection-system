# ============================================================
# app/model.py — Model Loading & Prediction Logic
# ============================================================
# 🧠 This file handles everything related to the ML model:
# - Loading the trained model from disk on startup
# - Loading the scaler for feature preprocessing
# - Making predictions on new transactions
# - Calculating risk scores and confidence levels

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS — Find model files relative to this file's location
# ============================================================
# 🧠 We use Path(__file__) so the paths work correctly whether
# you run the app from any directory. This is production best
# practice — never use hardcoded absolute paths like C:\Users\...
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# ============================================================
# GLOBAL VARIABLES — Loaded once when the app starts
# ============================================================
# 🧠 These are module-level variables. They are loaded ONCE
# when FastAPI starts and reused for every single prediction
# request. This is called "model caching" — loading a 5MB
# model file on every request would be extremely slow.
model = None
scaler = None
feature_names = None


def load_model():
    """
    Load the trained model, scaler and feature names from disk.
    Called once when FastAPI application starts up.
    Returns True if successful, False if any file is missing.
    """
    global model, scaler, feature_names

    try:
        if not MODEL_PATH.exists():
            print(f"❌ Model file not found at: {MODEL_PATH}")
            return False

        if not SCALER_PATH.exists():
            print(f"❌ Scaler file not found at: {SCALER_PATH}")
            return False

        if not FEATURE_NAMES_PATH.exists():
            print(f"❌ Feature names file not found at: {FEATURE_NAMES_PATH}")
            return False

        print("⏳ Loading fraud detection model...")
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded: {type(model).__name__}")

        print("⏳ Loading feature scaler...")
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded: {type(scaler).__name__}")

        print("⏳ Loading feature names...")
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        print(f"✅ Feature names loaded: {len(feature_names)} features")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def is_model_loaded():
    """Check if the model is loaded and ready for predictions."""
    return model is not None and scaler is not None and feature_names is not None


def predict_transaction(transaction_data: dict) -> dict:
    """
    Make a fraud prediction for a single transaction.

    Args:
        transaction_data: Dictionary with transaction features
                         (V1-V28, Amount, Time)

    Returns:
        Dictionary with prediction results including:
        - prediction (0 or 1)
        - prediction_label (Legitimate or Fraud)
        - fraud_probability (0.0 to 1.0)
        - confidence (LOW/MEDIUM/HIGH)
        - risk_score (0 to 100)
        - message (human readable result)
    """
    if not is_model_loaded():
        raise RuntimeError("Model is not loaded. Cannot make predictions.")

    # STEP 1 — Extract Amount and Time for scaling
    # 🧠 We must scale Amount and Time using the SAME scaler
    # from training to avoid training-serving skew
    amount = transaction_data.get("Amount", 0)
    time = transaction_data.get("Time", 0)

    # STEP 2 — Scale Amount and Time
    amount_scaled = scaler.transform([[amount]])[0][0]
    time_scaled = scaler.transform([[time]])[0][0]

    # STEP 3 — Build feature vector in exact same order as training
    # 🧠 Column ORDER matters in ML — wrong order = wrong prediction
    feature_vector = []
    for feature in feature_names:
        if feature == "Amount_Scaled":
            feature_vector.append(amount_scaled)
        elif feature == "Time_Scaled":
            feature_vector.append(time_scaled)
        else:
            feature_vector.append(transaction_data.get(feature, 0))

    # STEP 4 — Convert to DataFrame with correct column names
    input_df = pd.DataFrame([feature_vector], columns=feature_names)

    # STEP 5 — Make prediction
    prediction = int(model.predict(input_df)[0])
    fraud_probability = float(model.predict_proba(input_df)[0][1])

    # STEP 6 — Calculate risk score (0-100)
    # 🧠 This makes the API output more business-friendly.
    # A bank's fraud team understands "Risk Score: 87/100"
    # better than "fraud_probability: 0.87"
    risk_score = int(fraud_probability * 100)

    # STEP 7 — Determine confidence level
    # 🧠 Confidence tells the fraud analyst how certain the
    # model is. Low confidence predictions may need human review
    if fraud_probability < 0.3:
        confidence = "HIGH"
    elif fraud_probability < 0.6:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    # STEP 8 — Set confidence correctly based on prediction
    if prediction == 1:
        if fraud_probability >= 0.7:
            confidence = "HIGH"
            message = f"🚨 HIGH RISK: Transaction flagged as fraudulent with {fraud_probability*100:.1f}% confidence. Recommend immediate block."
        elif fraud_probability >= 0.4:
            confidence = "MEDIUM"
            message = f"⚠️ MEDIUM RISK: Transaction suspicious with {fraud_probability*100:.1f}% fraud probability. Recommend review."
        else:
            confidence = "LOW"
            message = f"⚠️ LOW RISK FLAG: Borderline transaction with {fraud_probability*100:.1f}% fraud probability. Monitor closely."
    else:
        if fraud_probability <= 0.1:
            confidence = "HIGH"
            message = f"✅ LOW RISK: Transaction appears legitimate with {(1-fraud_probability)*100:.1f}% confidence."
        elif fraud_probability <= 0.3:
            confidence = "MEDIUM"
            message = f"✅ LIKELY LEGITIMATE: Transaction probably safe but with some uncertainty ({fraud_probability*100:.1f}% fraud probability)."
        else:
            confidence = "LOW"
            message = f"⚠️ UNCERTAIN: Transaction classified as legitimate but with elevated fraud probability ({fraud_probability*100:.1f}%)."

    return {
        "prediction": prediction,
        "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(fraud_probability, 6),
        "confidence": confidence,
        "risk_score": risk_score,
        "message": message
    }