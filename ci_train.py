# ============================================================
# ci_train.py — Lightweight Model Training for CI/CD Pipeline
# ============================================================
# 🧠 This script runs on GitHub Actions during CI/CD.
# The real dataset (150MB CSV) is in .gitignore and never
# uploaded to GitHub. This script generates synthetic data
# that mimics the real dataset's statistical properties,
# trains a model quickly, and saves it for the test suite.
#
# This is a real MLOps pattern called "CI model training" —
# used when datasets are too large to store in version control.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 50)
print("CI/CD — Synthetic Model Training")
print("=" * 50)

# Set random seed for reproducibility
# 🧠 Same seed = same synthetic data every CI run
# This makes our tests deterministic — a key testing principle
np.random.seed(42)

# ============================================================
# GENERATE SYNTHETIC DATASET
# ============================================================
# 🧠 We generate data that mimics the statistical properties
# of the real credit card fraud dataset:
# - 99.83% legitimate transactions
# - 0.17% fraudulent transactions (severe class imbalance)
# - V1-V28 features follow normal distribution (like real PCA output)
# - Fraud transactions have distinctly different V feature patterns

N_LEGITIMATE = 5000
N_FRAUD = 50
N_FEATURES = 28

print(f"Generating {N_LEGITIMATE} legitimate transactions...")
legitimate = np.random.randn(N_LEGITIMATE, N_FEATURES)
legitimate_amount = np.abs(np.random.exponential(100, N_LEGITIMATE))
legitimate_time = np.sort(np.random.uniform(0, 172800, N_LEGITIMATE))
legitimate_labels = np.zeros(N_LEGITIMATE)

print(f"Generating {N_FRAUD} fraudulent transactions...")
# 🧠 Fraud transactions have shifted mean values on key features
# This mimics the real dataset where V14, V4, V12 etc. have
# very different distributions for fraud vs legitimate
fraud = np.random.randn(N_FRAUD, N_FEATURES)
fraud[:, 0] *= 3
fraud[:, 3] += 4
fraud[:, 6] -= 3
fraud[:, 9] -= 3
fraud[:, 11] -= 3
fraud[:, 13] -= 4
fraud[:, 16] -= 3
fraud_amount = np.abs(np.random.exponential(10, N_FRAUD))
fraud_time = np.random.uniform(0, 172800, N_FRAUD)
fraud_labels = np.ones(N_FRAUD)

# ============================================================
# BUILD DATAFRAME
# ============================================================
v_columns = [f"V{i}" for i in range(1, 29)]

legit_df = pd.DataFrame(legitimate, columns=v_columns)
legit_df["Amount"] = legitimate_amount
legit_df["Time"] = legitimate_time
legit_df["Class"] = legitimate_labels

fraud_df = pd.DataFrame(fraud, columns=v_columns)
fraud_df["Amount"] = fraud_amount
fraud_df["Time"] = fraud_time
fraud_df["Class"] = fraud_labels

df = pd.concat([legit_df, fraud_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total synthetic transactions: {len(df):,}")
print(f"Legitimate: {(df['Class']==0).sum():,}")
print(f"Fraudulent: {(df['Class']==1).sum():,}")

# ============================================================
# PREPROCESS
# ============================================================
scaler = StandardScaler()
df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_Scaled"] = scaler.fit_transform(df[["Time"]])
df = df.drop(["Amount", "Time"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

feature_names = X.columns.tolist()

# ============================================================
# TRAIN MODEL
# ============================================================
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
print("✅ Model trained successfully!")

# ============================================================
# SAVE MODEL FILES
# ============================================================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model.pkl")
print("✅ Model saved to models/fraud_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")
print("✅ Scaler saved to models/scaler.pkl")

joblib.dump(feature_names, "models/feature_names.pkl")
print("✅ Feature names saved to models/feature_names.pkl")

# ============================================================
# VERIFY
# ============================================================
loaded_model = joblib.load("models/fraud_model.pkl")
test_input = pd.DataFrame([X.iloc[0]], columns=feature_names)
pred = loaded_model.predict(test_input)[0]
print(f"✅ Sanity check prediction: {'Fraud' if pred == 1 else 'Legitimate'}")

print("=" * 50)
print("CI Model Training Complete!")
print("Models ready for test suite.")
print("=" * 50)