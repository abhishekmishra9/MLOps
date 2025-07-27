from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv("C:/Users/kalya/Downloads/DMML/mlops-housing/data/raw/housing.csv", index=False)

if __name__ == "__main__":
    load_data()