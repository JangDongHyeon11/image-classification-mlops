import pandas as pd
import tensorflow as tf
import imgaug.augmenters as iaa
from tensorflow.data import AUTOTUNE
from typing import List

# Augmentation configuration
augmenter = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.AddToBrightness((-30, 30))),
    iaa.Fliplr(1.0),
    iaa.OneOf([
        iaa.Affine(rotate=(-20, 20)),
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
    ])
], random_order=True)
AUGMENTER = iaa.Sometimes(0.8, augmenter)

# Function to load and preprocess images
def load_data(path, label, target_size):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    return image, tf.cast(label, tf.int32)

# Function to normalize images
def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

# Function to apply augmentation
def augment(image, label, augmenter):
    image = tf.numpy_function(augmenter.augment_image, [tf.cast(image, tf.uint8)], tf.uint8)
    return image, label

# Function to build the data pipeline
def build_data_pipeline(annot_df: pd.DataFrame, classes: List[str], split: str, img_size: List[int],
                        batch_size: int = 8, do_augment: bool = False, augmenter: iaa = None):
    df = annot_df[annot_df['split'] == split]
    paths = df['abs_path']
    labels = df[classes]
    
    pipeline = (tf.data.Dataset.from_tensor_slices((paths, labels))
                .shuffle(len(df))
                .map(lambda path, label: load_data(path, label, target_size=img_size), num_parallel_calls=AUTOTUNE)
                .map(normalize, num_parallel_calls=AUTOTUNE))
    
    if do_augment and augmenter:
        pipeline = pipeline.map(lambda x, y: augment(x, y, augmenter), num_parallel_calls=AUTOTUNE)
    
    return pipeline.batch(batch_size).prefetch(AUTOTUNE)
