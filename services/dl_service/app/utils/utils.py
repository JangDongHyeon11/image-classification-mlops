import os
import cv2
import base64
import logging
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer
from tensorflow.keras.models import load_model, Model


logger = logging.getLogger('main')


def array_to_encoded_str(image: np.ndarray):
    pil_img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='PNG', optimize = True)
    byte_data = img_buffer.getvalue()
    # compare to base64.b64encode(byte_data).decode('utf-8')
    img_str = base64.encodebytes(byte_data).decode("utf-8")
    return img_str
