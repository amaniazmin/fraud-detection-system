# ============================================================
# tests/test_api.py — Automated API Tests
# ============================================================
# 🧠 These tests use pytest + httpx to send real HTTP requests
# to our FastAPI app and verify the responses are correct.
# They run automatically in our GitHub Actions CI/CD pipeline.
# A passing test suite = the green ✅ badge on your GitHub repo.

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model import load_model

# ============================================================
# TEST CLIENT SETUP
# ============================================================
# 🧠 TestClient is a special client from FastAPI/Starlette that
# simulates HTTP requests WITHOUT needing a running server.
# It runs the app in-memory — much faster than real HTTP calls.
client = TestClient(app)

# ============================================================
# FIXTURES — Reusable test data
# ============================================================
# 🧠 A pytest fixture is a reusable piece of test data or setup.
# Instead of copying the same transaction dict into every test,
# we define it once here and inject it where needed.

@pytest.fixture
def legitimate_transaction():
    """A real legitimate transaction from the dataset."""
    return {
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


@pytest.fixture
def fraudulent_transaction():
    """A real fraudulent transaction from the dataset."""
    return {
        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.9979055875468,
        "V5": -0.522187864667764,
        "V6": -1.42654531920595,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927719433,
        "V10": -2.77227214465915,
        "V11": 3.20203320709635,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": 0.389724120274487,
        "V16": -1.14074717980657,
        "V17": -2.83005567450437,
        "V18": -0.0168224681808257,
        "V19": 0.416955705037907,
        "V20": 0.126910559061474,
        "V21": 0.517232370861764,
        "V22": -0.0350493686052974,
        "V23": -0.465211076182388,
        "V24": 0.320198198514526,
        "V25": 0.0445191674731724,
        "V26": 0.177839798284401,
        "V27": 0.261145002567677,
        "V28": -0.143275874698919,
        "Amount": 0.0,
        "Time": 406.0
    }


# ============================================================
# TEST GROUP 1 — General Endpoints
# ============================================================

def test_root_endpoint():
    """
    Test that the root endpoint returns 200 and correct keys.
    🧠 This verifies the API is running and responding.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    print("✅ test_root_endpoint passed")


def test_health_endpoint():
    """
    Test that the health endpoint returns healthy status.
    🧠 Cloud platforms call this to verify the service is up.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    assert data["version"] == "1.0.0"
    print("✅ test_health_endpoint passed")


# ============================================================
# TEST GROUP 2 — Prediction Endpoint
# ============================================================

def test_predict_legitimate_transaction(legitimate_transaction):
    """
    Test that a known legitimate transaction is correctly classified.
    🧠 This is a regression test — if the model changes and starts
    misclassifying this known transaction, the test will catch it.
    """
    response = client.post("/predict", json=legitimate_transaction)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["prediction_label"] == "Legitimate"
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "confidence" in data
    assert "message" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert 0 <= data["risk_score"] <= 100
    print(f"✅ test_predict_legitimate passed | Risk: {data['risk_score']}/100")


def test_predict_fraudulent_transaction(fraudulent_transaction):
    """
    Test that a known fraudulent transaction is correctly classified.
    🧠 This is our most critical test — we must catch real fraud.
    """
    response = client.post("/predict", json=fraudulent_transaction)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["prediction_label"] == "Fraud"
    assert data["fraud_probability"] > 0.5
    assert data["risk_score"] > 50
    print(f"✅ test_predict_fraudulent passed | Risk: {data['risk_score']}/100")


def test_predict_response_structure(legitimate_transaction):
    """
    Test that the prediction response has all required fields
    with correct data types.
    🧠 This is a contract test — it ensures our API response
    schema never breaks for downstream consumers.
    """
    response = client.post("/predict", json=legitimate_transaction)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["prediction"], int)
    assert isinstance(data["prediction_label"], str)
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["confidence"], str)
    assert isinstance(data["risk_score"], int)
    assert isinstance(data["message"], str)
    assert data["confidence"] in ["LOW", "MEDIUM", "HIGH"]
    print("✅ test_predict_response_structure passed")


def test_predict_invalid_data():
    """
    Test that sending invalid data returns a 422 validation error.
    🧠 FastAPI + Pydantic automatically validates input data.
    This test verifies that our schema correctly rejects bad input.
    """
    invalid_data = {
        "V1": "not_a_number",
        "Amount": -999,
        "Time": 0.0
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
    print("✅ test_predict_invalid_data passed")


def test_predict_missing_fields():
    """
    Test that sending incomplete data returns a 422 validation error.
    🧠 All 30 fields are required. Missing fields must be rejected.
    """
    incomplete_data = {
        "V1": 1.0,
        "Amount": 100.0
    }
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422
    print("✅ test_predict_missing_fields passed")


# ============================================================
# TEST GROUP 3 — Batch Prediction Endpoint
# ============================================================

def test_batch_predict(legitimate_transaction, fraudulent_transaction):
    """
    Test batch prediction with multiple transactions.
    🧠 Batch endpoints are used in production to process
    thousands of transactions efficiently in one API call.
    """
    batch = [legitimate_transaction, fraudulent_transaction]
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert "fraud_detected" in data
    assert "legitimate_detected" in data
    assert "fraud_rate" in data
    assert len(data["predictions"]) == 2
    print(f"✅ test_batch_predict passed | Fraud rate: {data['fraud_rate']}%")


def test_batch_predict_empty():
    """
    Test that an empty batch returns a 400 error.
    """
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400
    print("✅ test_batch_predict_empty passed")