# ============================================================
# app/main.py — FastAPI Application & Route Definitions
# ============================================================
# 🧠 This is the entry point of our API. FastAPI automatically:
# - Generates interactive docs at /docs (Swagger UI)
# - Generates alternative docs at /redoc
# - Validates all incoming requests using our schemas
# - Returns proper HTTP status codes and error messages

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from app.schemas import TransactionInput, PredictionOutput, HealthResponse
from app.model import load_model, predict_transaction, is_model_loaded

# ============================================================
# LIFESPAN — Runs on startup and shutdown
# ============================================================
# 🧠 The lifespan function replaces the old @app.on_event pattern.
# Code BEFORE yield runs on startup (load the model).
# Code AFTER yield runs on shutdown (cleanup if needed).
# This ensures the model is loaded before any request is handled.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("🚀 Starting Fraud Detection API...")
    print("=" * 50)
    success = load_model()
    if success:
        print("=" * 50)
        print("✅ API is ready to serve predictions!")
        print("📖 Interactive docs available at: /docs")
        print("=" * 50)
    else:
        print("❌ WARNING: Model failed to load!")
        print("Predictions will not work until model is available.")
    yield
    print("👋 Shutting down Fraud Detection API...")


# ============================================================
# APP INSTANCE — Create the FastAPI application
# ============================================================
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="""
## 🔍 Credit Card Fraud Detection System

A production-grade Machine Learning API that detects fraudulent 
credit card transactions in real-time.

### Features:
- **Real-time fraud detection** using Random Forest ML model
- **95.18% ROC-AUC score** trained on 284,807 real transactions
- **Risk scoring** from 0-100 for business decision making
- **Confidence levels** (LOW/MEDIUM/HIGH) for analyst triage
- **Auto-generated docs** with interactive testing interface

### How to use:
1. Send a POST request to `/predict` with transaction features
2. Receive instant fraud probability and risk score
3. Use the `/docs` page below to test interactively

### Model Performance:
- Precision: 91.86%
- Recall: 80.61%  
- F1 Score: 85.87%
- ROC-AUC: 95.18%
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================
# CORS MIDDLEWARE
# ============================================================
# 🧠 CORS (Cross-Origin Resource Sharing) allows our Streamlit
# dashboard (running on a different port/domain) to call this
# API. Without this, browsers block cross-origin requests for
# security reasons. allow_origins=["*"] allows ALL origins —
# fine for a portfolio project, restrict in real production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTE 1 — Root endpoint
# ============================================================
@app.get("/", tags=["General"])
async def root():
    """
    Welcome endpoint. Returns basic API information.
    """
    return {
        "message": "Welcome to the Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


# ============================================================
# ROUTE 2 — Health check endpoint
# ============================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"]
)
async def health_check():
    """
    Health check endpoint used by cloud platforms to verify
    the API is running and the model is loaded correctly.
    Returns 503 if model is not loaded.
    """
    model_ready = is_model_loaded()

    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0"
    )


# ============================================================
# ROUTE 3 — Main prediction endpoint
# ============================================================
@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict whether a credit card transaction is fraudulent.

    Accepts 30 transaction features (V1-V28, Amount, Time) and
    returns a fraud prediction with probability and risk score.

    - **prediction**: 0 = Legitimate, 1 = Fraud
    - **fraud_probability**: 0.0 to 1.0 probability of fraud
    - **risk_score**: 0 to 100 business-friendly risk score
    - **confidence**: LOW / MEDIUM / HIGH confidence level
    - **message**: Human readable explanation of the result
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Cannot process predictions."
        )

    try:
        start_time = time.time()
        transaction_dict = transaction.model_dump()
        result = predict_transaction(transaction_dict)
        processing_time = round((time.time() - start_time) * 1000, 2)
        print(f"✅ Prediction made in {processing_time}ms | Result: {result['prediction_label']} | Risk: {result['risk_score']}/100")
        return PredictionOutput(**result)

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# ROUTE 4 — Batch prediction endpoint
# ============================================================
@app.post(
    "/predict/batch",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK
)
async def predict_fraud_batch(transactions: list[TransactionInput]):
    """
    Predict fraud for multiple transactions at once.
    Accepts a list of transactions, returns a list of predictions.
    Maximum 100 transactions per batch request.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Cannot process predictions."
        )

    if len(transactions) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum of 100 transactions."
        )

    if len(transactions) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch cannot be empty."
        )

    try:
        results = []
        for i, transaction in enumerate(transactions):
            transaction_dict = transaction.model_dump()
            result = predict_transaction(transaction_dict)
            result["transaction_index"] = i
            results.append(result)

        fraud_count = sum(1 for r in results if r["prediction"] == 1)

        return {
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "legitimate_detected": len(results) - fraud_count,
            "fraud_rate": round(fraud_count / len(results) * 100, 2),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )