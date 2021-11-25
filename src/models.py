import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate, Convolution2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1

def create_first_model():
    """Initial model prototype that I used for testing in the notebook"""

    effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
    site1 = Input(shape=(224,224,3))
    site2 = Input(shape=(224,224,3))
    x = effnet(site1)
    x = GlobalAveragePooling2D()(x)
    x = Model(inputs=site1, outputs=x)
    y = effnet(site2)
    y = GlobalAveragePooling2D()(y)
    y = Model(inputs=site2, outputs=y)
    combined = concatenate([x.output, y.output])
    z = Dropout(0.5)(combined)
    z = Dense(1108, activation='softmax')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    model.summary()
    
    return model



def create_cnn_model(learning_rate=0.001):
    """
    CNN model
    """

    input_data = Input(name="input_image",shape=(6,224,224))

    cnn_seq = Sequential([
        Convolution2D(filters=64, kernel_size=(3,3), input_shape=(6,224,224), activation='relu', padding='same',data_format='channels_first'),
        Convolution2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Convolution2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
        Convolution2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
        Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
        Flatten(),
        Dense(units=8192, activation="relu"),
        Dense(units=4096, activation="relu"),
        Dense(units=1108, activation="softmax")
    ])

    cnn_model = cnn_seq(input_data)

    model = Model(inputs=[input_data], outputs=cnn_model)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    model.summary()

    return model