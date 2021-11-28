# %% [markdown]
# Using https://www.kaggle.com/cyannani123/keras-cellular-image-classification as test example

# %%
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1
from tensorflow.python.client import device_lib
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate, Convolution2D, MaxPool2D, Flatten
from tensorflow.keras.utils import Sequence
import os, sys, random,copy
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1

# %%
from tqdm import tqdm

# %%
sys.path.append('../src')

# %%
from data_v2 import augment
from data_v2 import get_center_box
from data_v2 import get_random_subbox
from data_v2 import ImgGen

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% [markdown]
# Loading test and train data.
# 
# Will load directly from zip for now.

# %%
train_data_all = pd.read_csv("../data/train.csv")
print("Shape of train_data_all:", train_data_all.shape)

train_data_all=train_data_all.loc[train_data_all.sirna.isin(["sirna_706","sirna_1046","sirna_586","sirna_747"]),:]

sirna_label_encoder_all = LabelEncoder().fit(train_data_all.sirna)
print(f"Sirna classes in train_data_all {len(sirna_label_encoder_all.classes_)}")

train_data_strat=train_data_all
#train_data_strat=train_data_all.groupby('sirna',group_keys=False).apply(lambda x: x.sample(frac=0.10).reset_index(drop=True))

train_data_sample_1 = train_data_strat.sample(frac=1,random_state=42).reset_index(drop=True)
print("Shape of train_data_sample_1:", train_data_sample_1.shape)
sirna_label_encoder_sample_1 = LabelEncoder().fit(train_data_sample_1.sirna)
print(f"Sirna classes in train_data_sample_1 {len(sirna_label_encoder_sample_1.classes_)}")


# %% [markdown]
# Saidid on samast wellist tehtud eri pildid. Pm võib võtta ainult ühe saidi sisse.
# Channelid on eri kanalitega tehtud pildid. Neid on kokku 6.

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate, Convolution2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers

# %%
from tensorflow import keras


conv_base = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(224, 224,3))
# %%
model = keras.Sequential(
    [
        keras.Input(shape=(6,224,224)),
        conv_base,
        layers.GlobalMaxPooling2D(name="gap"),
        layers.Dropout(0.5),
        layers.Dense(2000, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'),
        layers.Dense(len(sirna_label_encoder_sample_1.classes_), activation="softmax"),
    ]
)

conv_base.trainable = False

# %%
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# %%
model.summary()

# %% [markdown]
# Training model

# %%
test_size = 0.2
batch_size = 16

# %%
train, val = train_test_split(train_data_sample_1, test_size=test_size)

# %%
print(f"Training set size {len(train)}")
print(train.sirna.value_counts())
print(f"Validation set size {len(val)}")
print(val.sirna.value_counts())

# %%
train_gen = ImgGen(train,batch_size=batch_size,preprocess=get_center_box,shuffle=True,label_encoder=sirna_label_encoder_sample_1, path='../data/train/')
val_gen = ImgGen(val,batch_size=batch_size,preprocess=get_center_box,shuffle=True,label_encoder=sirna_label_encoder_sample_1, path='../data/train/')

# %%
print(f"Training set batched size {len(train_gen)}")
print(f"Validation set batched size {len(val_gen)}")

# %%
filepath = 'ModelCheckpoint_all.h5'

# %%
callback = [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        ]

# %%
history = model.fit(train_gen, 
                              steps_per_epoch=len(train)//batch_size, 
                              epochs=500, 
                              verbose=1, 
                              validation_data=val_gen,
                              validation_steps=len(val)//batch_size,
                              callbacks=callback
                             )

# plot history to png
import matplotlib.pyplot as plt


# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()


