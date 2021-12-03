import random

import tensorflow.keras as keras

import time


class CheckpointCallback(keras.callbacks.Callback):
    def __init__(self, run):
        self.run = run

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
        return

    def on_epoch_end(self, epoch, logs={}):
        self.run.log("Training accuracy", logs.get("accuracy"))
        self.run.log("Training loss", logs.get("loss"))
        self.run.log("Validation accuracy", logs.get("val_accuracy"))
        self.run.log("Validation loss", logs.get("val_loss"))

        epoch_time = time.time() - self.epoch_time_start

        self.run.log("Epoch time", epoch_time)

        return

    def on_batch_begin(self, batch, logs={}):

        self.batch_time_start = time.time()

        return

    def on_batch_end(self, batch, logs={}):

        batch_time = time.time() - self.batch_time_start

        self.run.log("Batch time", batch_time)

        return
