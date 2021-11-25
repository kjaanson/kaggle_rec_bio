#!/usr/bin/env python
# coding: utf-8

# Using https://www.kaggle.com/cyannani123/keras-cellular-image-classification as test example

# In[1]:


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


# In[2]:


from tqdm import tqdm


# In[3]:


sys.path.append('../src')


# In[4]:


from data_v2 import augment
from data_v2 import get_center_box
from data_v2 import get_random_subbox
from data_v2 import ImgGen


# In[5]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Loading test and train data.
# 
# Will load directly from zip for now.

# In[6]:


train_data_all = pd.read_csv("../data/train.csv")
print("Shape of train_data_all:", train_data_all.shape)
sirna_label_encoder_all = LabelEncoder().fit(train_data_all.sirna)
print(f"Sirna classes in train_data_all {len(sirna_label_encoder_all.classes_)}")


# In[10]:


train_data_all[['cell_line','exp']]=list(train_data_all.experiment.str.split('-'))


# Võtame esialgu ainul HUVEC liini

# In[18]:


train_data_all.loc[train_data_all.cell_line=='HUVEC',:]

train_data_strat=train_data_all.groupby('sirna',group_keys=False).apply(lambda x: x.sample(frac=0.25).reset_index(drop=True))



# In[56]:



train_data_sample = train_data_strat.sample(frac=1,random_state=42).reset_index(drop=True)
print("Shape of train_data_sample:", train_data_sample.shape)
sirna_label_encoder = LabelEncoder().fit(train_data_sample.sirna)
print(f"Sirna classes in train_data_all {len(sirna_label_encoder.classes_)}")


# Saidid on samast wellist tehtud eri pildid. Pm võib võtta ainult ühe saidi sisse.
# Channelid on eri kanalitega tehtud pildid. Neid on kokku 6.

# Training model

# In[52]:


test_size = 0.4
batch_size = 32


# In[53]:


train, val = train_test_split(train_data_sample, test_size=test_size, stratify=train_data_sample.sirna, random_state=42)


# In[59]:


print(f"Training set size {len(train)}")
print(f"Validation set size {len(val)}")


# In[61]:


train_gen = ImgGen(train,batch_size=batch_size,preprocess=get_center_box,shuffle=True,label_encoder=sirna_label_encoder, path='../data/train.zip')
val_gen = ImgGen(val,batch_size=batch_size,preprocess=get_center_box,shuffle=True,label_encoder=sirna_label_encoder, path='../data/train.zip')


# In[63]:


print("Preloading training data")
for i in tqdm(train_gen):
    pass

print("Preloading validation data")
for i in tqdm(val_gen):
    pass


# In[ ]:


print(f"Training set batched size {len(train_gen)}")
print(f"Validation set batched size {len(val_gen)}")


# In[ ]:


import joblib


# In[ ]:


joblib.dump(train_gen,'../data/train.pkl')


# In[ ]:


joblib.dump(val_gen,'../data/val.pkl')


# In[ ]:




