# Fraud Detection with MLflow

## Overview
This project uses MLflow to manage the lifecycle of a fraud detection model by using RandomForest, XGBoost, LogisticRegression with the `creditcard.csv` dataset which is taken from Kaggle. It includes training, tuning, deployment, and monitoring.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Start MLflow server: `mlflow server --host 127.0.0.1 --port 5000`
3. Run main script: `python main.py`
4. Start web server: `python src/serve_model.py` (access at `http://localhost:5001`)

## Files
- `main.py`: Runs the ML lifecycle
- `src/`: Contains model training, tuning, deployment, and serving code
- `src/templates/index.html`: Web interface
- `requirements.txt`: Dependencies

## Results
Best model and metrics logged in MLflow(check `http://localhost:5000`)

## Author
Beyza Bayrak , AIN-3009, Bahçeşehir University"# Development-and-Evaluation-of-a-ML-Lifecycle-Management-System-using-MLflow---Fraud-Detection" 
