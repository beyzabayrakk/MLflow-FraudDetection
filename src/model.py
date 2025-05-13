from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import importlib
import numpy as np

def log_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"conf_matrix_{model_name}.png")
    mlflow.log_artifact(f"conf_matrix_{model_name}.png")
    plt.close()

def train_and_log_model(name, cfg, params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        module = importlib.import_module(cfg["module"])
        model_class = getattr(module, cfg["class"])
        
        if name == "RandomForest":
            model = model_class(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            )
        elif name == "XGBoost":
            scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
            model = model_class(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight
            )
        elif name == "LogisticRegression":
            model = model_class(
                C=params['C'],
                max_iter=params['max_iter'],
                solver='liblinear',
                class_weight='balanced'
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

    
        mlflow.log_param("model_type", name)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log the model
        mlflow.sklearn.log_model(model, f"{name}_model")
        log_confusion_matrix(y_test, y_pred, name)

        print(f"{name} trained:\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nAUC: {auc:.4f}")
        return model, f1