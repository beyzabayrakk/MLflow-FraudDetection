import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def deploy_model(model, X_sample):
    with mlflow.start_run():
        signature = infer_signature(X_sample, model.predict(X_sample))
        mlflow.sklearn.log_model(
            model,
            "fraud_detection_model",
            signature=signature,
            registered_model_name="FraudDetectionModel"
        )
    print("Model is saved to MLflow Model Registry")
 