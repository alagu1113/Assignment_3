# utils/mlflow_utils.py
import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

def init_mlflow():
    # Use a valid local MLflow tracking URI
    tracking_uri = "file:///C:/Users/Alagu/EMI_Project/mlruns"
    mlflow.set_tracking_uri(tracking_uri)

def load_registered_model(model_uri):
    """
    Load a model from MLflow Registry or run.
    Example: 'models:/EMI_Classification_Model/Staging' or 'runs:/<run_id>/model'
    """
    init_mlflow()
    return mlflow.pyfunc.load_model(model_uri)

def list_experiments():
    """
    Return all MLflow experiments.
    Uses search_experiments() for compatibility with new MLflow versions.
    """
    init_mlflow()
    client = MlflowClient()

    # ✅ Updated function call for new MLflow versions
    try:
        experiments = client.search_experiments()
    except AttributeError:
        # backward compatibility (older MLflow versions)
        experiments = client.list_experiments()

    return experiments

def list_runs(experiment_name):
    """
    Return recent runs for a given experiment name.
    """
    init_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50
    )
    return runs

def register_prediction_log(run_name, model_name, input_data, pred, task="classification"):
    """
    Log prediction event details as a new MLflow run (optional auditing feature).
    """
    init_mlflow()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", task)
        mlflow.log_metric("prediction_count", 1)

        # Save input/output as JSON artifact
        import json, tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(json.dumps({"input": input_data, "prediction": pred}).encode())
        tmp.close()
        mlflow.log_artifact(tmp.name, artifact_path="prediction_logs")
