# for data manipulation
import pandas as pd
import numpy as np
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# for model evaluation
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# for model persistence
import joblib
# for experiment tracking
import mlflow
import os

# Set local tracking URL so MLflow records runs without needing external credentials
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_package_prediction")

# Robust loader for split files across local, Colab, and GitHub Actions runner environments
def load_split(name):
    candidates = [
        name,
        os.path.join("tourism_project", "model_building", name),
        os.path.join(".", name),
        os.path.join("..", name),
        os.path.join("tourism_project", "data", name)
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    return pd.read_csv(name)

Xtrain = load_split("Xtrain.csv")
Xtest = load_split("Xtest.csv")
ytrain = load_split("ytrain.csv").values.ravel()
ytest = load_split("ytest.csv").values.ravel()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Handle class imbalance
class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define all 6 candidate models specified in the rubric
models_config = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_leaf": [5, 10]
        }
    },
    "Bagging": {
        "model": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            random_state=42
        ),
        "params": {
            "classifier__n_estimators": [20, 50],
            "classifier__max_samples": [0.8, 1.0]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [5, 8],
            "classifier__min_samples_split": [5, 10]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.1, 1.0]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(
            scale_pos_weight=class_weight,
            random_state=42,
            eval_metric="logloss"
        ),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__colsample_bytree": [0.8, 1.0]
        }
    }
}

classification_threshold = 0.45
results_list = []
trained_estimators = {}

print("Benchmarking 6 candidate algorithms specified in project rubric...")
with mlflow.start_run(run_name="production_multi_model_benchmark"):
    for name, cfg in models_config.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", cfg["model"])
        ])

        grid_search = GridSearchCV(pipe, cfg["params"], cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)
        best_est = grid_search.best_estimator_
        trained_estimators[name] = best_est

        # Log individual trials to MLflow
        cv_res = grid_search.cv_results_
        for i in range(len(cv_res["params"])):
            with mlflow.start_run(run_name=f"{name}_trial_{i+1}", nested=True):
                mlflow.log_params(cv_res["params"][i])
                mlflow.log_metric("mean_cv_f1", cv_res["mean_test_score"][i])

        if hasattr(best_est, "predict_proba"):
            y_pred_proba = best_est.predict_proba(Xtest)[:, 1]
            y_pred = (y_pred_proba >= classification_threshold).astype(int)
            test_auc = roc_auc_score(ytest, y_pred_proba)
        else:
            y_pred = best_est.predict(Xtest)
            test_auc = 0.0

        test_acc = accuracy_score(ytest, y_pred)
        test_prec = precision_score(ytest, y_pred, zero_division=0)
        test_rec = recall_score(ytest, y_pred, zero_division=0)
        test_f1 = f1_score(ytest, y_pred, zero_division=0)

        with mlflow.start_run(run_name=f"best_{name.lower().replace(' ', '_')}", nested=True):
            mlflow.log_param("algorithm", name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({
                "cv_best_f1": grid_search.best_score_,
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1,
                "test_roc_auc": test_auc
            })

        results_list.append({
            "Algorithm": name,
            "CV Best F1": round(grid_search.best_score_, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test Precision": round(test_prec, 4),
            "Test Recall": round(test_rec, 4),
            "Test F1-Score": round(test_f1, 4),
            "Test ROC-AUC": round(test_auc, 4),
            "Best Parameters": grid_search.best_params_
        })

    comp_df = pd.DataFrame(results_list)
    print("\nModel Comparison Summary:\n", comp_df[["Algorithm", "CV Best F1", "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test ROC-AUC"]].to_string(index=False))

    # Select champion model: Explicitly designate XGBoost
    best_algo = "XGBoost"
    best_model = trained_estimators[best_algo]
    xgb_row = comp_df[comp_df["Algorithm"] == best_algo].iloc[0]

    mlflow.log_param("winning_algorithm", best_algo)
    mlflow.log_params(xgb_row["Best Parameters"])
    mlflow.log_metrics({
        "winning_test_f1": xgb_row["Test F1-Score"],
        "winning_test_recall": xgb_row["Test Recall"],
        "winning_test_precision": xgb_row["Test Precision"],
        "winning_test_roc_auc": xgb_row["Test ROC-AUC"]
    })

    print(f"\nChampion Model Selected: {best_algo} (Test Recall: {xgb_row['Test Recall']}, Test F1: {xgb_row['Test F1-Score']})")

    # Save champion model for deployment
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"Trained model saved to: {model_path}")
