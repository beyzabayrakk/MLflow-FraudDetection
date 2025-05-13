from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import importlib

def optimize_model(name, cfg, X_train, y_train, X_test, y_test):
    module = importlib.import_module(cfg["module"])
    model_class = getattr(module, cfg["class"])

    def rf_objective(params):
        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            random_state=42,
            class_weight='balanced_subsample', 
            n_jobs=1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        return {'loss': -f1, 'status': STATUS_OK}

    def xgb_objective(params):
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        return {'loss': -f1, 'status': STATUS_OK}

    def lr_objective(params):
        model = LogisticRegression(
            C=params['C'],
            max_iter=int(params['max_iter']),
            solver='liblinear',
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        return {'loss': -f1, 'status': STATUS_OK}

    search_spaces = {
        'RandomForest': {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 10),  
            'max_depth': hp.quniform('max_depth', 3, 20, 1)  
        },
        'XGBoost': {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 10),  
            'max_depth': hp.quniform('max_depth', 3, 15, 1),  
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3)  
        },
        'LogisticRegression': {
            'C': hp.loguniform('C', -3, 3),
            'max_iter': hp.quniform('max_iter', 100, 1000, 100)
        }
    }

    objectives = {
        'RandomForest': rf_objective,
        'XGBoost': xgb_objective,
        'LogisticRegression': lr_objective
    }

    # Hyperopt optimization 
    with mlflow.start_run(run_name=f"{name}_Hyperopt"):
        print(f"Optimizing {name}...")
        trials = Trials()
        best = fmin(
            fn=objectives[name],
            space=search_spaces[name],
            algo=tpe.suggest,
            max_evals=3,  
            trials=trials
        )

        # log the best parameters
        best_loss = min([trial['result']['loss'] for trial in trials.trials])
        mlflow.log_metric("best_f1_score", -best_loss)

        # log the details of each trial
        for i, trial in enumerate(trials.trials):
            mlflow.log_metric(f"trial_{i}_f1_score", -trial['result']['loss'])
            mlflow.log_param(f"trial_{i}_params", trial['misc']['vals'])

    
    if name == "RandomForest":
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': int(best['max_depth'])
        }
    elif name == "XGBoost":
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate']
        }
    elif name == "LogisticRegression":
        best_params = {
            'C': best['C'],
            'max_iter': int(best['max_iter'])
        }

    return best_params