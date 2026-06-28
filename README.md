# 💳 Credit Card Fraud Detection System

![CI/CD Pipeline](https://github.com/amaniazmin/fraud-detection-system/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A production-grade Machine Learning system that detects fraudulent credit card transactions in real-time, served via FastAPI REST API with a live Streamlit dashboard and automated CI/CD pipeline.

## 🔗 Live Links
| Resource | Link |
|---|---|
| 🌐 Live Dashboard | Coming soon |
| 📖 API Documentation | Coming soon |
| 📊 GitHub Actions | [CI/CD Pipeline](https://github.com/amaniazmin/fraud-detection-system/actions) |

## 🎯 Project Overview
Built an end-to-end fraud detection system trained on **284,807 real anonymized bank transactions** from European cardholders. The system detects fraudulent transactions in real-time with a **95.18% ROC-AUC score**.

## 📊 Model Performance
| Metric | Score |
|---|---|
| ROC-AUC | 95.18% |
| Precision | 91.86% |
| Recall | 80.61% |
| F1 Score | 85.87% |

## 🏗️ System Architecture
Credit Card Transaction

↓

Streamlit Dashboard (UI)

↓

FastAPI REST API (/predict)

↓

Random Forest ML Model

↓

Fraud / Legitimate Decision
## 🛠️ Tech Stack
| Layer | Technology | Purpose |
|---|---|---|
| ML Model | scikit-learn (Random Forest) | Fraud classification |
| API Backend | FastAPI + Uvicorn | REST API serving |
| Dashboard | Streamlit | Live web interface |
| Testing | pytest + httpx | Automated test suite |
| CI/CD | GitHub Actions | Automated testing pipeline |
| Deployment | Render + Streamlit Cloud | Cloud hosting |

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/amaniazmin/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python ci_train.py
```

### 5. Start FastAPI backend
```bash
uvicorn app.main:app --reload
```

### 6. Start Streamlit dashboard (new terminal)
```bash
streamlit run streamlit_app.py
```

### 7. Run tests
```bash
pytest tests/test_api.py -v
```

## 📁 Project Structure
fraud-detection-system/

├── app/

│   ├── main.py          # FastAPI routes & endpoints

│   ├── model.py         # ML model loading & prediction

│   └── schemas.py       # Pydantic data validation

├── models/

│   ├── fraud_model.pkl  # Trained Random Forest model

│   └── scaler.pkl       # Feature scaler

├── notebooks/

│   └── train_model.ipynb # Full model training notebook

├── tests/

│   └── test_api.py      # 9 automated API tests

├── .github/workflows/

│   └── ci.yml           # GitHub Actions CI/CD pipeline

├── streamlit_app.py     # Live dashboard

├── ci_train.py          # Lightweight CI training script

└── requirements.txt     # Project dependencies
## 👩‍💻 Author
**Amani Azmin**
- GitHub: [@amaniazmin](https://github.com/amaniazmin)

## 📄 License
MIT License