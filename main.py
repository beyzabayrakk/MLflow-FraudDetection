from src.data_preprocessing import load_data, clean_data
from src.model import train_and_log_model
from src.model_tuning import optimize_model
from src.deployment import deploy_model
from src.monitoring import monitor_model
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("FraudDetectionExperiment")

def main():
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test, scaler = clean_data(df)

    # Model
    models = {
        "RandomForest": {
            "class": "RandomForestClassifier",
            "module": "sklearn.ensemble"
        },
        "XGBoost": {
            "class": "XGBClassifier",
            "module": "xgboost"
        },
        "LogisticRegression": {
            "class": "LogisticRegression",
            "module": "sklearn.linear_model"
        }
    }

    best_model = None
    best_f1 = 0
    best_model_name = None

    for name, cfg in models.items():
        print(f"🔍 Optimizing {name}...")
        best_params = optimize_model(name, cfg, X_train, y_train, X_test, y_test)
        
        print(f"Training {name} with best parameters...")
        model, f1 = train_and_log_model(name, cfg, best_params, X_train, y_train, X_test, y_test)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"Best model selected: {best_model_name} with F1 score: {best_f1:.4f}")

    print("Deploying best model...")
    deploy_model(best_model, X_train)

    # production 
    client = MlflowClient()
    latest_version_info = client.get_latest_versions(name="FraudDetectionModel", stages=[])[-1]
    latest_version = latest_version_info.version

    client.transition_model_version_stage(
        name="FraudDetectionModel",
        version=latest_version,
        stage="Staging"
    )
    client.transition_model_version_stage(
        name="FraudDetectionModel",
        version=latest_version,
        stage="Production"
    )

    print("Monitoring model performance...")
    monitor_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()