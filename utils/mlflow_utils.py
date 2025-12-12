# utils/mlflow_utils.py

import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# MLflow Tracking Setup
# -----------------------------
TRACKING_URI = "file:///C:/Users/Alagu/EMI_Project/mlruns"
mlflow.set_tracking_uri(TRACKING_URI)

# -----------------------------
# Your MLflow Run IDs
# -----------------------------
RUN_ID_CLASSIFICATION = "67a2f4cfb9a54b6ba1c5902875ac322b"
RUN_ID_REGRESSION     = "8bce97d1297047efb7d0e86e47989e43"

# NOTE: If there are separate run folders inside the experiment,
# point to the exact model artifact folder inside each run.
MODEL_URI_CLASSIFICATION = f"runs:/{RUN_ID_CLASSIFICATION}/model"
MODEL_URI_REGRESSION     = f"runs:/{RUN_ID_REGRESSION}/model"


def load_classification_model():
    """Load EMI Eligibility Classification Model"""
    return mlflow.pyfunc.load_model(MODEL_URI_CLASSIFICATION)


def load_regression_model():
    """Load EMI Regression Model (Max EMI Prediction)"""
    return mlflow.pyfunc.load_model(MODEL_URI_REGRESSION)


def list_experiments():
    """Return all available MLflow experiments"""
    client = MlflowClient()
    try:
        return client.search_experiments()
    except:
        return client.list_experiments()


def list_runs(experiment_name):
    """Return recent runs for a given experiment"""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return []
    return client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50
    )


def register_prediction_log(run_name, model_name, input_data, pred, task="classification"):
    """Store prediction log into MLflow (optional for audit)"""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", task)
        mlflow.log_metric("prediction_count", 1)

        import json, tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(json.dumps({"input": input_data, "prediction": pred}).encode())
        tmp.close()
        mlflow.log_artifact(tmp.name, artifact_path="prediction_logs")
