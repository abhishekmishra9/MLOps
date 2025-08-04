# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 03:05:59 2025

@author: kalya
"""

import os
import joblib
import logging
import pandas as pd
from flask import Flask, request, jsonify, Response
from pydantic import BaseModel, ValidationError
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from sklearn.ensemble import RandomForestRegressor
from prometheus_flask_exporter import PrometheusMetrics

# Initialize Flask app
app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info('ml_api', 'ML Housing API', version='1.0.0')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total number of requests")
PREDICTION_COUNT =Counter("prediction_requests_total", "Total prediction requests")
ERROR_COUNT = Counter("prediction_errors_total", "Total prediction errors")
RETRAIN_COUNT = Counter("retrain_requests_total", "Total retrain requests")

# Model path
MODEL_PATH = os.path.join("models", "best_model", "model.pkl")

# Input schema

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Load model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


model = load_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": " Housing Price Prediction API (Flask)"}), 200



@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.inc()

    if model is None:
        ERROR_COUNT.inc()
        return jsonify({"error": "Model not available."}), 500

    try:
        input_json = request.get_json()
        validated_input = HousingInput(**input_json)
        input_df = pd.DataFrame([validated_input.dict()])
        prediction = model.predict(input_df)[0]
        logging.info(f"Prediction made|Input:{input_json}| Output:{prediction}")
        PREDICTION_COUNT.inc()
        return jsonify({"prediction": round(float(prediction), 4)}), 200

    except ValidationError as ve:
        ERROR_COUNT.inc()
        return jsonify({"error": "Invalid input", "details": ve.errors()}), 400

    except Exception as e:
        ERROR_COUNT.inc()
        logging.exception(" Prediction error") 
        print(e)
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/retrain", methods=["POST"])
def retrain():
    RETRAIN_COUNT.inc()
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Load CSV
        df = pd.read_csv(file)
        required_columns = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude', 'target'
        ]

        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing columns in uploaded data."}), 400

        X = df[required_columns[:-1]]
        y = df['target']

        # Retrain model
        new_model = RandomForestRegressor()
        new_model.fit(X, y)

        # Save and reload
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)
        global model
        model = new_model

        logging.info(f"Model retrained with {len(df)} records and saved.")
        return jsonify({"message":f"Model retrained on {len(df)} samples."}), 200

    except Exception as e:
        ERROR_COUNT.inc()
        print(e)
        logging.exception("Retrain error")
        return jsonify({"error": "Retrain failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
