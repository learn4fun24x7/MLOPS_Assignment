# MLOPS Assignment

## Scenario
Building a minimal but complete MLOps pipeline for a ML model using California Housing open dataset Kaggle. The model needs to be trained, tracked, versioned, deployed as an API, and monitored for prediction usage.

## Learning Outcomes
- Used Git, DVC, and MLflow for versioning and tracking.
- Packaged ML code as a REST API using FastAPI.
- Containerized and deployed it using Docker.
- Configured a GitHub Actions pipeline for CI/CD.
- Implemented basic logging and exposed monitoring metrics.

## Technologies
- Git + GitHub
- DVC
- MLflow
- Docker
- Flask or FastAPI
- GitHub Actions
- Logging module with Prometheus/Grafana

## Assignment Tasks

### Part 1: Repository and Data Versioning (4 marks)
- Set up a GitHub repo.
- Load and preprocess the dataset.
- Track the dataset with DVC (if using California Housing).
- Maintain clean directory structure.

### Part 2: Model Development & Experiment Tracking
- Train at least two models (Linear Regression, Decision Tree for Housing).
- Use MLflow to track experiments (params, metrics, models).
- Select best model and register in MLflow.

### Part 3: API & Docker Packaging
- Create an API for prediction using Flask or FastAPI.
- Containerize the service using Docker.
- Accept input via JSON and return model prediction.

### Part 4: CI/CD with GitHub Actions
- Lint/test code on push.
- Build Docker image and push to Docker Hub.
- Deploy locally or to EC2/LocalStack using shell script or docker run.

### Part 5: Logging and Monitoring
- Log incoming prediction requests and model outputs.
- Store logs to file or simple in-memory DB (SQLite).
- Optionally, expose /metrics endpoint for monitoring.
 
### Bonus
- Add input validation using pydantic or schema.
- Integrate with Prometheus and create a sample dashboard.
- Add model re-training trigger on new data.
