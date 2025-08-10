# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    
    # Load raw data
    df = pd.read_csv(input_path)

    print("Dataset loaded successfully. Displaying first 5 rows:")
    print(df.head())

    print("\nDataset Info:")
    df.info()
    
    print("\nMissing values before pre-processing:")
    print(df.isnull().sum())

    # Basic cleaning (California Housing usually has no missing values)
    df.dropna(inplace=True)

    print("\nMissing values after post-processing:")
    print(df.isnull().sum())

    # Split features and target
    X = df.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)
    y = df["median_house_value"]

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"\nNumerical features: {list(num_cols)}")
    print(f"Categorical features: {list(cat_cols)}")

    # Create preprocessing pipelines for numerical and categorical features
    # Numerical pipeline: Impute missing values with the median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values with the most frequent value, then One-Hot Encode
    # 'ocean_proximity' is the only categorical feature in this dataset.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' allows unseen categories during transform
    ])

    # Combine preprocessing steps using ColumnTransformer
    # This applies different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("\nFitting preprocessor on training data and transforming both training and test data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if cat_cols != []:   
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
        processed_feature_names = list(num_cols) + list(ohe_feature_names)
    else:
        processed_feature_names = list(num_cols)

    # Convert processed arrays back to DataFrames for easier inspection and further use
    X_train_processed = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)

    print("\nPreprocessing complete. Displaying first 5 rows of processed training data:")
    print(X_train_processed.head())

    print(f"\nShape of processed training data: {X_train_processed.shape}")
    print(f"Shape of processed testing data: {X_test_processed.shape}")
    print(f"Shape of training target: {y_train.shape}")
    print(f"Shape of testing target: {y_test.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data to CSV files
    print(f"Saving processed data to CSV files in '{output_dir}'...")
    X_train_processed.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_processed.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("Data successfully saved!")

if __name__ == '__main__':
    preprocess_data("data/raw/housing.csv", "data/processed")