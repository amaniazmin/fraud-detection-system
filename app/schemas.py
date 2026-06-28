# ============================================================
# app/schemas.py — Data Validation Schemas
# ============================================================
# 🧠 Pydantic schemas define the shape of data coming IN and
# going OUT of our API. FastAPI uses these automatically to:
# 1. Validate incoming requests (reject bad data instantly)
# 2. Generate automatic API documentation at /docs
# 3. Serialize outgoing responses to JSON

from pydantic import BaseModel, Field
from typing import Optional


class TransactionInput(BaseModel):
    """
    Schema for a single credit card transaction.
    These are the 30 features our model expects.
    V1-V28 are PCA-transformed anonymized features.
    """
    V1: float = Field(..., description="PCA transformed feature 1")
    V2: float = Field(..., description="PCA transformed feature 2")
    V3: float = Field(..., description="PCA transformed feature 3")
    V4: float = Field(..., description="PCA transformed feature 4")
    V5: float = Field(..., description="PCA transformed feature 5")
    V6: float = Field(..., description="PCA transformed feature 6")
    V7: float = Field(..., description="PCA transformed feature 7")
    V8: float = Field(..., description="PCA transformed feature 8")
    V9: float = Field(..., description="PCA transformed feature 9")
    V10: float = Field(..., description="PCA transformed feature 10")
    V11: float = Field(..., description="PCA transformed feature 11")
    V12: float = Field(..., description="PCA transformed feature 12")
    V13: float = Field(..., description="PCA transformed feature 13")
    V14: float = Field(..., description="PCA transformed feature 14")
    V15: float = Field(..., description="PCA transformed feature 15")
    V16: float = Field(..., description="PCA transformed feature 16")
    V17: float = Field(..., description="PCA transformed feature 17")
    V18: float = Field(..., description="PCA transformed feature 18")
    V19: float = Field(..., description="PCA transformed feature 19")
    V20: float = Field(..., description="PCA transformed feature 20")
    V21: float = Field(..., description="PCA transformed feature 21")
    V22: float = Field(..., description="PCA transformed feature 22")
    V23: float = Field(..., description="PCA transformed feature 23")
    V24: float = Field(..., description="PCA transformed feature 24")
    V25: float = Field(..., description="PCA transformed feature 25")
    V26: float = Field(..., description="PCA transformed feature 26")
    V27: float = Field(..., description="PCA transformed feature 27")
    V28: float = Field(..., description="PCA transformed feature 28")
    Amount: float = Field(..., description="Transaction amount in euros", ge=0)
    Time: float = Field(..., description="Seconds elapsed since first transaction", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Amount": 149.62,
                "Time": 0.0
            }
        }


class PredictionOutput(BaseModel):
    """
    Schema for what our API returns after prediction.
    """
    prediction: int = Field(..., description="0=Legitimate, 1=Fraud")
    prediction_label: str = Field(..., description="Human readable label")
    fraud_probability: float = Field(..., description="Probability of fraud 0.0-1.0")
    confidence: str = Field(..., description="Confidence level: LOW/MEDIUM/HIGH")
    risk_score: int = Field(..., description="Risk score 0-100")
    message: str = Field(..., description="Human readable result message")


class HealthResponse(BaseModel):
    """
    Schema for the health check endpoint.
    🧠 Health checks are used by cloud platforms to verify
    your API is running correctly. Standard in production.
    """
    status: str
    model_loaded: bool
    version: str