import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        rmse, mae, r2 = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return rmse, run.info.run_id

def register_best_model(run_id, model_name="CaliforniaHousingModel"):
    
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Register model
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException:
        pass  # Model already exists

    model_version = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)

    # Transition to staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    print(f"Registered {model_name} version {model_version.version} and moved to Staging.")

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = load_data()

    rmse_lr, run_id_lr = train_and_log_model("LinearRegression", LinearRegression(), X_train, y_train, X_test, y_test)
    rmse_dt, run_id_dt = train_and_log_model("DecisionTree", DecisionTreeRegressor(), X_train, y_train, X_test, y_test)

    if rmse_lr < rmse_dt:
        best_run_id = run_id_lr
        best_model_name = "LinearRegression"
    else:
        best_run_id = run_id_dt
        best_model_name = "DecisionTree"

    print(f"Best model: {best_model_name}")
    register_best_model(best_run_id)
