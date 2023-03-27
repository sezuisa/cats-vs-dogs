
from keras.models import load_model
from keras.utils import load_img, img_to_array, plot_model
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


def VGG16():

    model = Sequential()

    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding='same', activation='relu'))

    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='vgg16'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(256, activation='relu', name='fc1'))

    model.add(Dense(128, activation='relu', name='fc2'))

    model.add(Dense(196, activation='softmax', name='output'))

    return model

# visualise model
plot_model(
    VGG16(),
    to_file='model_vgg16.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)