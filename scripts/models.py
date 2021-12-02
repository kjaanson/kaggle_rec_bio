import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate, Convolution2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from tensorflow.keras import layers
from tensorflow import keras

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


from tensorflow.keras.applications.inception_v3 import InceptionV3
def create_inception(learning_rate=0.0001, nr_classes=1108, input_shape=(6,224,224)):
    input_tensor = Input(shape=input_shape)

    layer_1 = layers.Conv2D(10, (1, 1), activation='relu', padding='same', data_format='channels_first')(input_tensor)
    layer_1 = layers.Conv2D(10, (3, 3), activation='relu', padding='same', data_format='channels_first')(layer_1)

    layer_2 = layers.Conv2D(10, (1, 1), activation='relu', padding='same', data_format='channels_first')(input_tensor)
    layer_2 = layers.Conv2D(10, (5, 5), activation='relu', padding='same', data_format='channels_first')(layer_2)

    layer_3 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1,1), padding='same', data_format='channels_first')(input_tensor)
    layer_3 = layers.Conv2D(10, (1, 1), activation='relu', padding='same', data_format='channels_first')(layer_3)

    mid = layers.concatenate([layer_1, layer_2, layer_3], axis=3)

    flat_1 = layers.Flatten()(mid)
    dense_1 = layers.Dense(256, activation='relu')(flat_1)
    dense_1 = layers.Dropout(0.5)(dense_1)
    dense_2 = layers.Dense(128, activation='relu')(dense_1)
    dense_2 = layers.Dropout(0.5)(dense_2)
    dense_3 = layers.Dense(64, activation='relu')(dense_2)
    dense_3 = layers.Dropout(0.5)(dense_3)
    dense_4 = layers.Dense(nr_classes, activation='softmax')(dense_3)

    model = Model(inputs=input_tensor, outputs=dense_4)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    model.summary()

    return model





def create_cnn_model_2(learning_rate=0.00002, nr_classes=1108):
    """
    CNN model based on latest archidecture in prototype
    """

    model = keras.Sequential(
        [
            keras.Input(shape=(6,224,224)),
            layers.Conv2D(64, kernel_size=(1, 1), activation="relu", padding="same", input_shape=(6,224,224), data_format="channels_first"),
            layers.Conv2D(64, kernel_size=(4, 4), activation="relu", padding="same", data_format="channels_first"),
            layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"),
            layers.AveragePooling2D(pool_size=(2, 2), data_format="channels_first"),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"),
            #layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.AveragePooling2D(pool_size=(3, 3), data_format="channels_first"),
            layers.SpatialDropout2D(0.5, data_format="channels_first"),
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(nr_classes, activation="softmax"),
        ]
    )
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
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