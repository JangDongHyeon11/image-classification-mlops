import os
import shutil
import mlflow
import requests
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Union, Tuple, Any
from prefect import task, get_run_logger, variables
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer, GlobalAveragePooling2D
from .utils.tf_data_utils import build_data_pipeline

PREFECT_PORT = os.getenv('PREFECT_PORT', '4200')
PREFECT_API_URL = os.getenv('PREFECT_API_URL',f'http://prefect:{PREFECT_PORT}/api')

@task(name='deploy_prefect_flow', log_prints=True)
def deploy_prefect_flow(git_repo_root: str, deploy_name: str):
    subprocess.run([f"cd {git_repo_root} && prefect --no-prompt deploy --name {deploy_name}"],
                    shell=True)

@task(name='create_or_update_prefect_vars')
def create_or_update_prefect_vars(kv_vars: Dict[str, Any]):
    logger = get_run_logger()
    for var_name, var_value in kv_vars.items():
        headers = {'Content-type': 'application/json'}
        body = {
                "name": var_name,
                "value": var_value
                }
        current_value = variables.get(var_name)
        if current_value is None:
            # create if not exist
            logger.info(f"Creating a new variable: {var_name}={var_value}")
            url = f'{PREFECT_API_URL}/variables'
            res = requests.post(url, json=body, headers=headers)
            if not str(res.status_code).startswith('2'):
                logger.error(f'Failed to create a Prefect variable, POST return {res.status_code}')
            logger.info(f'status code: {res.status_code}')
            
        else:
            # update if already existed
            logger.info(f"The variable '{var_name}' has already existed, updating the value with '{var_value}'")
            url = f'{PREFECT_API_URL}/variables/name/{var_name}'
            res = requests.patch(url, json=body, headers=headers)
            if not str(res.status_code).startswith('2'):
                logger.error(f'Failed to create a Prefect variable, PATCH return {res.status_code}')
            logger.info(f'status code: {res.status_code}')

@task(name='put_model_to_service')
def put_model_to_service(cfg: dict, service_host: str='nginx',
                        service_port: str='80'):
    logger = get_run_logger()
    endpoint = f'http://{service_host}:{service_port}/update_model/{cfg}'
    res = requests.put(endpoint)
    if res.status_code == 200:
        logger.info("PUT model to the service successfully")
    else:
        logger.error("PUT model failed")
        raise Exception(f"Failed to put model to {endpoint}")
