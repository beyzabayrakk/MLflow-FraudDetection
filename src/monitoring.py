import mlflow
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# set up logging
logging.basicConfig(level=logging.INFO)

#  MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

def monitor_model(model, X_test, y_test):
    with mlflow.start_run():
        logging.info("Started MLflow run.")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_param("model_type", model.__class__.__name__)  
        mlflow.log_metric("monitored_precision", precision)
        mlflow.log_metric("monitored_recall", recall)
        mlflow.log_metric("monitored_f1", f1)
        mlflow.log_metric("monitored_auc", auc)

        logging.info(f"Logged metrics → Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")