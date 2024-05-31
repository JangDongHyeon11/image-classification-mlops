import os
import mlflow
import pandas as pd
from importlib import import_module
import ray
from ray import train
from ray.train import Trainer
import hydra
from omegaconf import DictConfig
from jobs.model import initialize_model, save_model, train_model, upload_model,build_drift_model
from jobs.dataset import prepare_data_loader, validate_data
from jobs.deploy import (build_ref_data, save_and_upload_ref_data,
                           save_and_upload_drift_detectors)
from jobs.utils.tf_data_utils import AUGMENTER
from flows.utils import log_mlflow_info, build_and_log_mlflow_url
from prefect import flow, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import Dict, Any

STORAGE_CL_PATH = os.getenv("CENTRAL_STORAGE_PATH", "/home/Jang/central_storage")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@flow(name='train_flow')
def train_flow(cfg: DictConfig):
    logger = get_run_logger()
    central_ref_data_dir = os.path.join(STORAGE_CL_PATH, 'ref_data')
    hparams = cfg.train.hparams
    ds_cfg = cfg.dataset
    mlflow_train_cfg = cfg.train.mlflow
    model_cfg = cfg.model
    drift_cfg = model_cfg.drift_detection
    input_shape = (model_cfg.input_size.h, model_cfg.input_size.w)
    num_workers = cfg.train.num_workers
    artifact_path = cfg.artifact_path
    # 모델 구축
    model = initialize_model(input_size=input_shape, n_classes=len(model_cfg.classes),  
                        activation=model_cfg.activation,
                        layer=model_cfg.layer)

    # 데이터 불러오기
    repository_path, dataset_annotation_df  = prepare_data_loader(dataset_root=ds_cfg.ds_root, 
                                                  dataset_name=ds_cfg.ds_name, 
                                                  dvc_tag=ds_cfg.dvc_tag, 
                                                  dvc_checkout=ds_cfg.dvc_checkout)

    # 데이터 유효성 검사
    report_path = f"{ds_cfg.ds_root}/{ds_cfg.ds_name}_{ds_cfg.dvc_tag}_validation.html"
    validate_data(repository_path, img_ext='jpeg', save_path=report_path)
    
    
    mlflow.set_experiment(mlflow_train_cfg.exp_name)
    with mlflow.start_run(description=mlflow_train_cfg.exp_desc) as train_run:
        log_mlflow_info(logger, train_run)
        run_id = train_run.info.run_id
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mlflow_run_url = build_and_log_mlflow_url(logger, train_run)
        
        mlflow.set_tags(tags=mlflow_train_cfg.exp_tags)
        mlflow.log_params(hparams)
        mlflow.log_artifact(report_path)
        
        trainer = Trainer(backend="tensorflow", num_workers=num_workers)
        trained_model = trainer.run(train_model.remote,
                                    model, model_cfg.classes, repository_path, dataset_annotation_df,
                                    img_size=input_shape, epochs=hparams.epochs,
                                    batch_size=hparams.batch_size, init_lr=hparams.init_lr,
                                    augmenter=AUGMENTER)
        

        model_dir, metadata_file_path = save_model(trained_model, model_cfg,model_uri)
        
        upload_model(model_uri=model_uri, 
                    model_name=model_cfg.model_name,
                    MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI
                    )
        
        # 드리프트 모델 구축
        uae, bbsd = build_drift_model(trained_model, model_input_size=input_shape,
                                          softmax_layer_idx=drift_cfg.bbsd_layer_idx,
                                          encoding_dims=drift_cfg.uae_encoding_dims)
    
        save_and_upload_drift_detectors(uae, bbsd, model_cfg=model_cfg)

        ref_data_df = build_ref_data(uae, bbsd, dataset_annotation_df, n_sample=drift_cfg.reference_data_n_sample,
                                     classes=model_cfg.classes, img_size=input_shape, batch_size=hparams.batch_size)
        
        save_and_upload_ref_data(ref_data_df, central_ref_data_dir, model_cfg) 

    create_link_artifact(
        key = 'mlflow-train-run',
        link = mlflow_run_url,
        description = "Link to MLflow's training run"
    )

    return model_save_dir, metadata_file_path, metadata_file_name

@hydra.main(config_path="configs/", config_name="train_config.yaml")
def start(cfg: DictConfig):
    ray.init()  # Ray 초기화
    train_flow(cfg)
    ray.shutdown()   
if __name__ == "__main__":
    start()
