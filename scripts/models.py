import tensorflow as tf


import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate, Convolution2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

def create_cnn_model(learning_rate=0.001):

    input_data_1 = Input(name="input_image_c1_c3",shape=(224,224,3))
    input_data_2 = Input(name="input_image_c4_c6",shape=(224,224,3))

    input_data = keras.layers.Concatenate()([input_data_1, input_data_2])

    cnn_seq = Sequential([
        Convolution2D(filters=64, kernel_size=(3,3), input_shape=(224,224,6), activation='relu', padding='same',data_format='channels_last'),
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

    model = Model(inputs=[input_data_1, input_data_2], outputs=cnn_model)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    model.summary()

    return model