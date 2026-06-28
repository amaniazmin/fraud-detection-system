# conftest.py
# 🧠 This file is automatically loaded by pytest before any tests run.
# It sets up the Python path AND ensures the ML model is loaded
# into memory before the TestClient makes any requests.

import sys
import os

# Add project root to Python path so "from app.main import app" works
sys.path.insert(0, os.path.dirname(__file__))

# Load the model before any tests run
from app.model import load_model

def pytest_configure(config):
    """
    pytest_configure runs before test collection.
    We load the model here so it's ready for all tests.
    """
    print("\n⏳ Loading model for tests...")
    success = load_model()
    if success:
        print("✅ Model loaded for tests successfully!")
    else:
        print("❌ Model failed to load for tests!")