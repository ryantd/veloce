from tensorflow.keras.callbacks import Callback
import ray.train as train


class TFTrainReportCallback(Callback):
    def on_epoch_end(self, _, logs=None):
        train.report(**logs)
