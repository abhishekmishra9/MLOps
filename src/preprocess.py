import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import os


def load_data():
    # Fetch dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Save raw data
    raw_path = "C:/Users/kalya/Downloads/DMML/mlops-housing/data/raw/housing.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")
    return df


def preprocess_data(df):
    # Example preprocessing: scale all numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop("MedHouseVal", axis=1))

    # Create a new DataFrame with scaled features
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
    df_scaled["MedHouseVal"] = df["MedHouseVal"]

    # Save processed data
    processed_path = "C:/Users/kalya/Downloads/DMML/mlops-housing/data/processed/housing.csv"
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_scaled.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

    return df_scaled


if __name__ == "__main__":
    raw_df = load_data()
    processed_df = preprocess_data(raw_df)
