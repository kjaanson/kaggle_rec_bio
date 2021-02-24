import tensorflow as tf


import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

def create_cnn_model(learning_rate=0.001):

    input_data_1 = Input(name="input_image_c1_c3",shape=(224,224,3))
    input_data_2 = Input(name="input_image_c4_c6",shape=(224,224,3))

    input_data = keras.layers.Concatenate()([input_data_1, input_data_2])

    conv_3d = keras.layers.Conv2D(
        filters=16, 
        kernel_size=(3,3), 
        padding='same', 
        name='c3d',
        data_format='channels_last')(input_data)

    activation = keras.layers.Activation('relu')(conv_3d)


    pooling = keras.layers.MaxPooling2D(pool_size=(4,4))(activation)

    flatten = tf.keras.layers.Flatten()(pooling)

    dense1 = Dense(2048)(flatten)

    act2 = keras.layers.Activation('relu')(dense1)

    dropout = Dropout(0.25)(act2)

    dense = Dense(1108, activation='softmax')(dropout)

    model = Model(inputs=[input_data_1, input_data_2], outputs=dense)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    model.summary()

    return model