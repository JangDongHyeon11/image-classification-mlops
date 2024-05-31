import os
import time
import pandas as pd
from typing import List
from dvc.repo import Repo
from git import Git, GitCommandError
from prefect import task, get_run_logger
from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import train_test_validation

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