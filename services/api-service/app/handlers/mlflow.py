import os
import mlflow
from typing import Tuple
from pprint import pprint
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel


class MLflowHandler:
    def __init__(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
    
    def check_mlflow_health(self) -> None:
        try:
            experiments = mlflow.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return "Service returning experiments"
        except:
            return "Error calling MLflow"  
    
    def get_production_model(self, model_name: str) -> Tuple[PyFuncModel, str, str]:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
        latest_versions_metadata = self.client.get_latest_versions(name=model_name)
        model_version = latest_versions_metadata[0].version
        return model, model_name, model_version