import os
import yaml
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
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



MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow_client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def build_model_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return pd.DataFrame(report).T

# def build_model_metadata(model_cfg): 
#     metadata = model_cfg.copy()
#     metadata.pop('save_dir')
#     return metadata

def build_model_metadata(model_cfg): 
    metadata = OmegaConf.to_container(model_cfg, resolve=True)
    metadata.pop('save_dir')
    return metadata

def build_figure_from_df(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = pd.plotting.table(ax, df, loc='center', cellLoc='center')  # where df is your data frame
    plt.show()
    return fig, table



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
    mlflow.tensorflow.log_model(model, artifact_path="model")
    
    return model_dir, metadata_save_path

@task(name='upload_model')
def upload_model(model_uri: str,model_dir: str,MLFLOW_TRACKING_URI: str , metadata_file_path: str):
    model_name = os.path.split(model_dir)[-1]
    metadata_file_name = os.path.split(metadata_file_path)[-1]
    logger = get_run_logger()
    
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered model")

    # transition model to production
    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
    )
    logger.info("Transitioned model to Staging stage")
    return metadata_file_name,model_name
    
@task(name='load_model')
def load_model(model_name: str):
    logger = get_run_logger()
    logger.info(f'Loading the model from {model_name}')
    
    model = mlflow.tensorflow.load_model(model_uri=f"models:/{model_name}/4/model")
    # model = mlflow.pyfunc.load_model(model_uri=f"runs:/bb1787ec35e742148e70296009ad8536/classifier_model_v1")
    
    latest_versions_metadata = mlflow_client.get_latest_versions(name=model_name)
    model_version = latest_versions_metadata[0].version
    # latest_model_version_metadata = mlflow_client.get_model_version(
    #     name=model_name, version=model_version
    # )
    logger.info('Loaded successfully')
    logger.info(f'model_name: {model_name} model_version: {model_version}')
    return model,model_version


@task(name='evaluate_model')
def evaluate_model(model: tf.keras.models.Model, classes: List[str], ds_repo_path: str, 
                   annotation_df: pd.DataFrame, subset: str, img_size: List[int], classifier_type: str='multi-class', 
                   multilabel_thr: float=0.5):
    logger = get_run_logger()
    logger.info(f"Building a data pipeline from '{subset}' set")
    test_ds = build_data_pipeline(annotation_df, classes, subset, img_size,
                                   do_augment=False, augmenter=None)
    logger.info('Getting ground truths and making predictions')
    y_true_bin = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds)
    if classifier_type == 'multi-class':
        y_true = np.argmax(y_true_bin, axis=1)
        y_pred = tf.argmax(y_pred_prob, axis=1)
    else: # multi-label
        y_true = y_true_bin
        y_pred = (y_pred_prob > multilabel_thr).astype(np.int8)

    if classifier_type == 'multi-class':
        # confusion matrix 생성
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Plot the confusion matrix 
        conf_matrix_fig = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True,
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix')
        plt.show()
        
        #  AUC
        roc_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
        
        # classification report
        report = build_model_report(y_true, y_pred, classes)
        
    elif classifier_type == 'multi-label':
        conf_matrix_fig = None
        roc_auc = roc_auc_score(y_true, y_pred_prob, average=None, multi_class='ovr')
        
        # Print classification report
        report = build_model_report(y_true, y_pred, classes)
        report['AUC'] = list(roc_auc) + (4*[None])
    logger.info('Log output to MLflow to complete the process')
    if conf_matrix_fig:
        mlflow.log_figure(conf_matrix_fig, 'confusion_matrix.png')
    if isinstance(roc_auc, float):
        mlflow.log_metric("AUC", roc_auc)
    # log_figure is a lot easier to look at from ui than log_table
    report = report.apply(lambda x: round(x, 5))
    report = report.reset_index()
    report_fig, _ = build_figure_from_df(report)
    mlflow.log_figure(report_fig, 'model_report.png')
    mlflow.log_table(report, 'model_report.json')

    
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


@task(name='save_drift_model')
def save_drift_model(uae_model: tf.keras.models.Model, bbsd_model: tf.keras.models.Model,
                                   model_cfg: Dict[str, Union[str, List[str], List[int]]]):
    logger = get_run_logger()
    uae_model_dir = os.path.join(model_cfg.save_dir, model_cfg.model_name + model_cfg.drift_detection.uae_model_suffix)
    if not os.path.exists(model_cfg.save_dir):
        logger.info(f"save_dir {model_cfg.save_dir} does not exist. Created.")
        os.makedirs(model_cfg.save_dir)
    uae_model.save(uae_model_dir)
    logger.info(f"Untrained AutoEncoder (UAE) model for {model_cfg.model_name} is saved to {uae_model_dir}")

    bbsd_model_dir = os.path.join(model_cfg.save_dir, model_cfg.model_name + model_cfg.drift_detection.bbsd_model_suffix)
    if not os.path.exists(model_cfg.save_dir):
        logger.info(f"save_dir {model_cfg.save_dir} does not exist. Created.")
        os.path.makedirs(model_cfg.save_dir)
    bbsd_model.save(bbsd_model_dir)
    logger.info(f"Black-Box Shift Detector (BBSD) model for {model_cfg.model_name} is saved to {bbsd_model_dir}")
    
    # upload to mlflow
    mlflow.log_artifact(uae_model_dir)
    mlflow.log_artifact(bbsd_model_dir)



