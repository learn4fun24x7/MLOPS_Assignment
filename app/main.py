# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging
import joblib

# Setup logging
logging.basicConfig(filename="logs/predictions.log", level=logging.INFO)

# Define input schema
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

# Load model from MLflow Model from local
model = joblib.load("model/CaliforniaHousingModel.pkl")

'''
# Load model from MLflow Model Registry
model_name = "CaliforniaHousingModel"
model_stage = "Staging"  # or "Production"
# mlflow.set_tracking_uri("http://127.0.0.1:5000") # for Running this API in VM
# mlflow.set_tracking_uri("host.docker.internal:5000") # for Running this API in docker
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")
'''

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "California Housing API"}

@app.post("/predict")
def predict(data: HousingInput):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    # Log request and prediction
    logging.info(f"Input: {input_dict}, Prediction: {prediction}")

    return {"prediction": prediction}
