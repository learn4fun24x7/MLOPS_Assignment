# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    
    # Load raw data
    df = pd.read_csv(input_path)

    # Basic cleaning (California Housing usually has no missing values)
    df.dropna(inplace=True)

    # Split features and target
    X = df.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)
    y = df["median_house_value"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    preprocess_data("data/raw/housing.csv", "data/processed")