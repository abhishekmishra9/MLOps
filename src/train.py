# train.py
import mlflow
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import mlflow.sklearn

# Set MLflow tracking and experiment
data = "file:///C:/Users/kalya/Downloads/DMML/mlops-housing/mlruns"
mlflow.set_tracking_uri(data)
mlflow.set_experiment("CaliforniaHousing")

# Load dataset
file = "C:/Users/kalya/Downloads/DMML/mlops-housing/data/raw/housing.csv"
df = pd.read_csv(file)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Candidate models
models = [
    ("LinearRegression", LinearRegression()),
    ("DecisionTree", DecisionTreeRegressor())
]

best_mse = float("inf")
best_model = None
best_model_name = ""
best_preds = None

# Train and evaluate models
for model_name, model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"{model_name} MSE: {mse:.4f}")
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_model_name = model_name
        best_preds = preds

# Start MLflow run and log best model
with mlflow.start_run(run_name="best_model") as run:
    signature = infer_signature(X_train, best_preds)
    input_example = X_train.iloc[:1]

    # Log to MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="CaliforniaHousingModel",
        input_example=input_example,
        signature=signature)

    mlflow.log_param("model_name", best_model_name)
    mlflow.log_metric("mse", best_mse)

    print(f"\n Best model '{best_model_name}' logged with MSE={best_mse:.4f}")

    # Save locally in MLflow format for API
    save_path = "models/best_model"
    os.makedirs(save_path, exist_ok=True)
    mlflow.sklearn.save_model(
        sk_model=best_model,
        path=save_path,
        input_example=input_example,
        signature=signature
    )
    print(f"Model saved locally for API at: {save_path}")
