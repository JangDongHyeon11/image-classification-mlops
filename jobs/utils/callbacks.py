import mlflow
from tensorflow.keras.callbacks import Callback

class MLflowLog(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(logs, step=epoch)