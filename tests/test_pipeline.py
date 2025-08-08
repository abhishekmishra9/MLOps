import os
import pandas as pd
import joblib
import pytest
from src import preprocess
from app import app, MODEL_PATH


# ---------- Test Preprocessing ----------
def test_preprocess_output():
    df_raw = preprocess.load_data()
    df_processed = preprocess.preprocess_data(df_raw)

    # Check dataframe shape and columns
    assert isinstance(df_processed, pd.DataFrame)
    assert "MedHouseVal" in df_processed.columns
    assert df_processed.isnull().sum().sum() == 0  # no missing values


# ---------- Test Training ----------
def test_training_creates_model():
    # Run training script
    os.system("python src/train.py")
    assert os.path.exists(MODEL_PATH), "Model file not found after training"

    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model does not have predict()"


# ---------- Test API Endpoints ----------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Housing Price Prediction API" in response.get_data(as_text=True)


def test_predict_endpoint(client):
    if not os.path.exists(MODEL_PATH):
        os.system("python src/train.py")

    payload = {
        "MedInc": 5.1,
        "HouseAge": 20.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.0,
        "Population": 800.0,
        "AveOccup": 2.5,
        "Latitude": 34.2,
        "Longitude": -118.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.get_json()
