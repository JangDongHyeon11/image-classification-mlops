import os
import time
import pandas as pd
import mlflow
import tensorflow as tf
import numpy as np
from typing import List, Dict, Union, Tuple, Any
from dvc.repo import Repo
from git import Git, GitCommandError
from prefect import task, get_run_logger
from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import train_test_validation
from .utils.tf_data import build_data_pipeline

@task(name='prepare_data_loader')
def prepare_data_loader(dataset_root: str, dataset_name: str, dvc_tag: str, dvc_checkout: bool = True):
    logger = get_run_logger()
    logger.info("데이터셋 name: {} | DvC tag: {}".format(dataset_name, dvc_tag))
    dataset_path = os.path.join(dataset_root, dataset_name)

    at_path = os.path.join(dataset_path, 'annotation_df.csv')
    dataset_annotation_df = pd.read_csv(at_path)
    
    return dataset_path, dataset_annotation_df

@task(name='validate_data')
def validate_data(dataset_path: str, save_path: str = 'dataset_val.html', img_ext: str = 'jpeg'):
    logger = get_run_logger()
    train_dataset, test_dataset = classification_dataset_from_directory(
        root=os.path.join(dataset_path, 'images'), object_type='VisionData',
        image_extension=img_ext
    )
    suite = train_test_validation()
    logger.info("데이터 검증 테스트 실행 중")
    result_dataset = suite.run(train_dataset, test_dataset)
    result_dataset.save_as_html(save_path)
    logger.info(f'데이터 검증을 완료하고 보고서를 다음 위치에 저장합니다. {save_path}')
    logger.info("이 파일은 이후 단계에서 MLflow의 학습 작업과 함께 저장됩니다.")
    
@task(name='build_ref_data')
def build_ref_data(uae_model: tf.keras.models.Model, bbsd_model: tf.keras.models.Model, 
                   annotation_df: pd.DataFrame, n_sample: int, classes: List[str], 
                   img_size: List[int], batch_size: int):
    logger = get_run_logger()
    train_ds = build_data_pipeline(annotation_df, classes, 'train', img_size, batch_size, 
                                   do_augment=False, augmenter=None)

    sampled_train_ds = train_ds.take(n_sample)
    logger.info('Getting ground truths and extracting features')
    y_true_bin = np.concatenate([y for _, y in sampled_train_ds], axis=0)
    uae_feats = uae_model.predict(sampled_train_ds)
    bbsd_feats = bbsd_model.predict(sampled_train_ds)
    data = {
        'uae_feats': list(uae_feats),
        'bbsd_feats': list(bbsd_feats),
        'label': list(y_true_bin)
    }
    ref_data_df = pd.DataFrame(data)
    return ref_data_df


@task(name='save_ref_data')
def save_ref_data(ref_data_df: pd.DataFrame, remote_dir: str, 
                             model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    logger = get_run_logger()
    save_file_name = model_cfg['model_name'] + model_cfg['drift_detection']['reference_data_suffix'] + '.parquet'
    save_file_path = os.path.join(model_cfg['save_dir'], save_file_name)
    if not os.path.exists(model_cfg['save_dir']):
        logger.info(f"save_dir {model_cfg['save_dir']} does not exist. Created.")
        os.makedirs(model_cfg['save_dir'])
    ref_data_df.to_parquet(save_file_path)
    logger.info(f'Saved ref_data in {save_file_path}')
    
    mlflow.log_artifact(save_file_path)

