import os
import yaml
import shutil
import ray
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from typing import List, Dict, Union, Tuple, Any
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from prefect import task, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import List, Dict, Union
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from .utils.tf_data import build_data_pipeline
from .utils.callbacks import MLflowLog

def build_model_metadata(model_cfg): 
    metadata = model_cfg.copy()
    metadata.pop('save_dir')
    return metadata

@task(name='initialize_model')
def initialize_model(input_size: list, n_classes: int, activation: str = 'softmax',
                layer: str = 'classify'):
    logger = get_run_logger()
    backbone = ResNet50(include_top=False, weights='imagenet',
                         input_shape = [input_size[0], input_size[1], 3])
    
    # ResNet50의 output에 추가 layer들을 붙임
    x = GlobalAveragePooling2D()(backbone.output)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(n_classes, activation=activation, name=layer)(x)
    model = Model(inputs=backbone.input, outputs=x)
    # 모델 요약 출력
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    logger.info("Model summary:")
    logger.info('\n'.join(summary))

    return model

@ray.remote
@task(name='train_model')
def train_model(model: tf.keras.models.Model, classes: List[str], repository_path: str, 
                dataset_annotation_df: pd.DataFrame, img_size: List[int], epochs: int, batch_size: int, 
                init_lr: float, augmenter: iaa):
    logger = get_run_logger()
    logger.info('Building data pipelines')
    
    train_ds = build_data_pipeline(dataset_annotation_df, classes, 'train', img_size, batch_size, 
                                   do_augment=True, augmenter=augmenter)
    valid_ds = build_data_pipeline(dataset_annotation_df, classes, 'valid', img_size, batch_size, 
                                   do_augment=False, augmenter=None)
    
    # compile
    opt = Adam(learning_rate=init_lr)
    loss = CategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    # callbacks
    mlflow_log = MLflowLog()
    
    # fit
    logger.info('Start training')
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=epochs,
              callbacks=[mlflow_log]
             )
    
    # return trained model
    return model


@task(name='save_model')
def save_model(model: tf.keras.models.Model, model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    logger = get_run_logger()
    
    model_dir = os.path.join(model_cfg.save_dir, model_cfg.model_name)
    if not os.path.exists(model_cfg.save_dir):
        logger.info(f"save_dir {model_cfg.save_dir} does not exist. Created.")
        os.makedirs(model_cfg.save_dir)
    model.save(model_dir)
    logger.info(f'Model is saved to {model_dir}')

    model_metadata = build_model_metadata(model_cfg)
    metadata_save_path = os.path.join(model_cfg.save_dir, model_cfg.model_name+'.yaml')
    with open(metadata_save_path, 'w') as f:
        yaml.dump(model_metadata, f)
    
    mlflow.log_artifact(model_dir)
    mlflow.log_artifact(metadata_save_path)
    
    return model_dir, metadata_save_path

@task(name='upload_model')
def upload_model(model_uri: str, model_name: str,MLFLOW_TRACKING_URI: str):
    mlflow_client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    logger = get_run_logger()
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered model")

    # transition model to production
    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="production",
    )
    logger.info("Transitioned model to production stage")
    
@task(name='build_drift_model')
def build_drift_model(main_model: tf.keras.models.Model, model_input_size: Tuple[int, int], 
                          softmax_layer_idx: int = -1, encoding_dims: int = 32):
    uae = tf.keras.Sequential(
        [
            InputLayer(input_shape = model_input_size+(3,)),
            Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
            Flatten(),
            Dense(encoding_dims,)
        ]
    )
    bbsd = Model(inputs=main_model.inputs, outputs=[main_model.layers[softmax_layer_idx].output])

    return uae, bbsd


