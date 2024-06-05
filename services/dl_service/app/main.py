import os
import cv2
import base64
import contextlib
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from utils import (array_to_encoded_str,  
                   prepare_db, commit_results_to_db, 
                   commit_only_api_log_to_db, check_db_healthy)
from handlers.mlflow import MLflowHandler
handlers = {}

class MetaData(BaseModel):
    model_name: str
    uae_name: str
    bbsd_name: str

class Message(BaseModel):
    message: str

class PredictionResult(BaseModel):
    model_name: str
    prediction: Dict[str,float]
    overlaid_img: str
    raw_hm_img: str
    message: str
    
FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)-3d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(FORMATTER)
logger.addHandler(ch)

# prepare database
prepare_db()


async def get_service_handlers():
    global handlers
    mlflow_handler = MLflowHandler()
    handlers["mlflow"] = mlflow_handler
    logging.info("Retrieving mlflow handler...")
    
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await get_service_handlers()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health_check",response_model=Message, responses={404: {"model": Message}})
async def health_check():
    resp_code = 200
    resp_message = "Service is ready and healthy."
    try:
        check_db_healthy()
        handlers["mlflow"].check_mlflow_health()
    except:
        resp_code = 404
        resp_message = "DB is not functional. Service is unhealthy."
    return JSONResponse(status_code=resp_code, content={"message": resp_message})

@app.put("/update_model", response_model=Message, responses={404: {"model": Message}})
def update_model(request: Request, meta_data: MetaData, background_tasks: BackgroundTasks):
    global model
    global model_meta
    global uae
    global bbsd
    start_time = time.time()
    logger.info('Updating model')
    try:
        model, model_name, model_version = handlers["mlflow"].get_production_model(
            model_name=meta_data.model_name
        )
        uae, model_name, model_version = handlers["mlflow"].get_production_model(
            model_name=meta_data.uae
        )
        bbsd, model_name, model_version = handlers["mlflow"].get_production_model(
            model_name=meta_data.bbsd
        )
        
    except Exception as e:
        logger.error(f'Loading model failed with exception:\n {e}')
        time_spent = round(time.time() - start_time, 4)
        resp_code = 404
        resp_message = f"Updating model failed"
        background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
        return JSONResponse(status_code=resp_code, content={"message": resp_message})
    
    time_spent = round(time.time() - start_time, 4)
    resp_code = 200
    resp_message = "Update the model successfully"
    background_tasks.add_task(commit_only_api_log_to_db, request, resp_code, resp_message, time_spent)
    return {"message": resp_message}

