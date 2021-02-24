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

# Helper class for real-time logging
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
        self.run.log('Training accuracy', logs.get('accuracy'))
        self.run.log('Training loss', logs.get('loss'))
        self.run.log('Validation accuracy', logs.get('val_accuracy'))
        self.run.log('Validation loss', logs.get('val_loss'))
        
        epoch_time=time.time() - self.epoch_time_start
        
        self.run.log('Epoch time', epoch_time)
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, dest='data_file', help='data folder mounting point')
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--train-frac', type=float, default=1.0, help='fraction of training data to take in')

    args = parser.parse_args()
    
    data_file = args.data_file
    epochs = args.epochs

    learning_rate = 0.001

    print("============================================")
    
    print("Input dataset: " + data_file)
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    os.makedirs('./outputs', exist_ok=True)
    
    archive = zipfile.ZipFile(data_file,'r')
    
    test_data = pd.read_csv(BytesIO(archive.read('test.csv')))
    print("Shape of test_data:", test_data.shape)
    test_data.head()
    
    train_data = pd.read_csv(BytesIO(archive.read('train.csv')))
    print("Shape of train_data:", train_data.shape)
    train_data.head()
    
    sirna_label_encoder = LabelEncoder().fit(train_data.sirna)
    
    joblib.dump(sirna_label_encoder, './outputs/sirna_label_encoder.joblib')
    
    @retry(tries=3)
    def get_input(experiment, plate, well, site, channel, train=True):
        if train==True:
            base_path = 'train'
        else:
            base_path = 'test'

        try:
            path = f"{base_path}/{experiment}/Plate{plate}/{well}_s{str(site)}_w{str(channel)}.png"
            img = Image.open(BytesIO(archive.read(path)))
        except KeyError as err:
            print(f"Error loading input - {err}")
            print("Will default to other site")

            # hack mis aitab kahe puuduva pildi puhul
            # pm kui puudub pilt siis proovib lihtsalt teist saiti võtta
            if site==2:
                path = f"{base_path}/{experiment}/Plate{plate}/{well}_s1_w{str(channel)}.png"
                img = Image.open(BytesIO(archive.read(path)))
            else:
                path = f"{base_path}/{experiment}/Plate{plate}/{well}_s2_w{str(channel)}.png"
                img = Image.open(BytesIO(archive.read(path)))

        imgr = img.resize( (224,224) )
    
        return imgr
    
    class ImgGen(Sequence):
        def __init__(self, label_data, batch_size = 32, preprocess=(lambda x: x), shuffle=False):

            if shuffle:
                self.label_data=label_data.sample(frac=1).reset_index(drop=True)
            else:
                self.label_data=label_data

            self.batch_size=batch_size
            self.preprocess=preprocess

        def __len__(self):
            return int(np.ceil(len(self.label_data))/float(self.batch_size))

        def __getitem__(self, i):

            batch_x = self.label_data.loc[i*self.batch_size:(i+1)*self.batch_size,("experiment","plate","well")]
            batch_y = self.label_data.loc[i*self.batch_size:(i+1)*self.batch_size,("sirna")]

            x_s1_c1 = [np.array(get_input(e, p, w, site=1, channel=1))/255 for e, p, w in batch_x.values.tolist()]
            x_s1_c2 = [np.array(get_input(e, p, w, site=1, channel=2))/255 for e, p, w in batch_x.values.tolist()]
            x_s1_c3 = [np.array(get_input(e, p, w, site=1, channel=3))/255 for e, p, w in batch_x.values.tolist()]

            x_s2_c1 = [np.array(get_input(e, p, w, site=1, channel=4))/255 for e, p, w in batch_x.values.tolist()]
            x_s2_c2 = [np.array(get_input(e, p, w, site=1, channel=5))/255 for e, p, w in batch_x.values.tolist()]
            x_s2_c3 = [np.array(get_input(e, p, w, site=1, channel=6))/255 for e, p, w in batch_x.values.tolist()]

            x1 = np.array([x_s1_c1,x_s1_c2,x_s1_c3]).transpose((1,2,3,0))
            x2 = np.array([x_s2_c1,x_s2_c2,x_s2_c3]).transpose((1,2,3,0))

            y = sirna_label_encoder.transform(batch_y)

            return [np.array(x1), np.array(x2)], y
        
        
    def augment(image):
        random_transform = random.randint(-1,4)
        if random_transform==0:
            image = image.rotate(random.randint(-5,5))
        if random_transform==1:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        if random_transform==2:
            image = image.filter(ImageFilter.RankFilter(size=3, rank=1))
        if random_transform==3:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        if random_transform==4:
            image = image.filter(ImageFilter.MaxFilter(size=3))
        return image
    

    run = Run.get_submitted_run()
    
    model = models.create_cnn_model()
    
    test_size = 0.025
    batch_size = 8
    
    run.log('Batch Size', batch_size)
    run.log('Test fraction', test_size)
    run.log('Training samples', len(train_data))
    run.log('Learning rate', learning_rate)
    
    aml_callback = CheckpointCallback(run)
    
    #resampling entire training dataset
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    train, val = train_test_split(train_data, test_size=test_size)
    
    print(f"Training set size {len(train)}")
    print(f"Validation set size {len(val)}")
    
    train_gen = ImgGen(train,batch_size=batch_size,shuffle=True)
    val_gen = ImgGen(val,batch_size=batch_size,shuffle=True)
    
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
    
    