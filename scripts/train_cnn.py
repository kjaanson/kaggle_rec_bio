import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications import EfficientNetB0

import os, sys, random,copy
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter
from sklearn.model_selection import train_test_split

import zipfile
from io import StringIO, BytesIO

from sklearn.preprocessing import LabelEncoder

from azureml.core import Run

from retry import retry

import argparse

import time
import joblib

import models

from helpers import CheckpointCallback

from data_new import ImgGen, get_center_box


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, dest='data_path', help='data folder mounting point')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch', type=int, help='number of epochs to train')
    parser.add_argument('--train-frac', type=float, default=1.0, dest='train_frac', help='fraction of training data to take in')

    args = parser.parse_args()
    
    data_path = args.data_path
    epochs = args.epochs
    training_fraction = args.train_frac

    learning_rate = 0.001

    print("============================================")
    
    print("Input dataset: " + data_path)
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    os.makedirs('./outputs', exist_ok=True)
    
    test_data = pd.read_csv(f"{data_path}/test.csv")
    print("Shape of test_data:", test_data.shape)
    test_data.head()
    
    train_data = pd.read_csv(f"{data_path}/train.csv")
    print("Shape of train_data:", train_data.shape)
    train_data.head()
    
    sirna_label_encoder = LabelEncoder().fit(train_data.sirna)
    
    joblib.dump(sirna_label_encoder, './outputs/sirna_label_encoder.joblib')    

    run = Run.get_submitted_run()
    
    model = models.create_cnn_model()
    
    test_size = 0.025
    batch_size = args.batch
    
    run.log('Batch Size', batch_size)
    run.log('Test fraction', test_size)
    run.log('Training samples', len(train_data))
    run.log('Learning rate', learning_rate)
    
    aml_callback = CheckpointCallback(run)
    
    #resampling entire training dataset
    train_data = train_data.sample(frac=training_fraction).reset_index(drop=True)

    train, val = train_test_split(train_data, test_size=test_size)
    
    print(f"Training set size {len(train)}")
    print(f"Validation set size {len(val)}")
    
    train_gen = ImgGen(train, label_encoder=sirna_label_encoder, path=data_path, batch_size=batch_size, preprocess=get_center_box, shuffle=True)
    val_gen = ImgGen(val, label_encoder=sirna_label_encoder, path=data_path, batch_size=batch_size, preprocess=get_center_box, shuffle=True)
    
    print(f"Training set batched size {len(train_gen)}")
    print(f"Validation set batched size {len(val_gen)}")
    
    
    filepath = './outputs/ModelCheckpoint_all.h5'
    
    callback = [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        ]
    
    history = model.fit(train_gen, 
                        steps_per_epoch=len(train)//batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        validation_steps=len(val)//batch_size,
                        callbacks=[callback, aml_callback]
                        )
    
    