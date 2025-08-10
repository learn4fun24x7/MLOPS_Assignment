# California Housing Price Prediction API
This project exposes a regression machine learning model for predicting California housing prices using FastAPI, running inside a Docker container.

The model is trained on the California housing dataset from Kaggle.

## Features
- Predict median house value based on input features
- REST API built with FastAPI
- Containerized using Docker for easy deployment

## How to Run
- docker pull learn4fun24x7/california-housing-api
- docker run -d -p 8000:8000 learn4fun24x7/california-housing-api