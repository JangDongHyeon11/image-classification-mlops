import os
import yaml
import mlflow
import pandas as pd
from jobs.model import evaluate_model, load_model
from jobs.dataset import prepare_data_loader
from workflows.utils import log_mlflow_info, build_and_log_mlflow_url
from prefect import flow, get_run_logger
from omegaconf import DictConfig
from prefect.artifacts import create_link_artifact
from typing import Dict, Any


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@flow(name='eval_flow')
def eval_flow(cfg: DictConfig):
  
    logger = get_run_logger()
    eval_cfg = cfg.eval
    model_cfg= cfg.model
    model_name= cfg.model.model_name
    metadata_file_path= cfg.model.model_metadata_file_path
    mlflow_eval_cfg = eval_cfg.mlflow
    ds_cfg = cfg.dataset
    

    model,model_version=load_model(model_name)
    # with open(metadata_file_path,'r') as f:
    #     model_cfg = yaml.safe_load(f)
    
    input_shape = (model_cfg.input_size.h, model_cfg.input_size.w)
    # 데이터 불러오기
    repository_path, dataset_annotation_df  = prepare_data_loader(dataset_root=ds_cfg.ds_root, 
                                                  dataset_name=ds_cfg.ds_name, 
                                                  dvc_tag=ds_cfg.dvc_tag, 
                                                  dvc_checkout=ds_cfg.dvc_checkout)
    mlflow.set_experiment(mlflow_eval_cfg.exp_name)
    with mlflow.start_run(description=mlflow_eval_cfg.exp_desc) as eval_run:
        log_mlflow_info(logger, eval_run)
        eval_run_url = build_and_log_mlflow_url(logger, eval_run)
        # Store model config
        # mlflow.log_artifact(metadata_file_path)
        evaluate_model(model, model_cfg.classes, repository_path, dataset_annotation_df,
                       subset=eval_cfg.subset, img_size=input_shape, classifier_type=model_cfg.classifier_type)       
        
    create_link_artifact(
        key = 'mlflow-evaluate-run',
        link = eval_run_url,
        description = "Link to MLflow's evaluation run"       
    )
    
def start(cfg: DictConfig):
    
    eval_flow(cfg)
   
if __name__ == "__main__":
    start()
  
    
    
    
        