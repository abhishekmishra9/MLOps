import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "housing.csv"
)

PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "housing.csv"
)


def load_data():
    """Fetch California housing dataset and save raw CSV."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw data saved to {RAW_DATA_PATH}")
    return df


def preprocess_data(df):
    """Scale numerical features and save processed CSV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop("MedHouseVal", axis=1))

    df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
    df_scaled["MedHouseVal"] = df["MedHouseVal"]

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_scaled.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

    return df_scaled


if __name__ == "__main__":
    raw_df = load_data()
    preprocess_data(raw_df)
