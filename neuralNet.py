import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import matplotlib.image as mpimg
from datetime import datetime
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import time

#---------------------------------
# CONSTANTS
EPOCHS = 10
RUNS = 10
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
BASE_DIR = 'data'
MODEL_NO = '3-2'
SEED_NUM = 1
DEFAULT_OPT = optimizers.Adam(learning_rate=0.001)

# ----------- Prepare datasets
def prepare_datasets():
    # create directories
    dataset_home = 'dataset/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['dogs/', 'cats/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)

    # seed random number generator
    seed(SEED_NUM)

    # define ratio of pictures to use for validation
    val_ratio = 0.2

    # copy training dataset images into subdirectories
    src_directory = 'original_dataset/'
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/'  + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/'  + file
            copyfile(src, dst)

# ----------- Define the model
def define_model(model_no):
    model = Sequential()
    # Model 1: VGG-1
    if model_no == '1':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))

    # Model 2: VGG-2
    elif model_no == '2':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))

    # Model 3: VGG-3 (with Image Data Augmentation for Model 3-3)
    elif model_no == '3' or model_no == '3-3':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))

    # Model 3-2: VGG-3 with Dropout
    elif model_no == '3-2':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

    # Model 3-4: VGG-3 with Dropout and Image Data Augmentation
    elif model_no == '3-4':
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

    else:
        raise ValueError('Undefined model number!')

	# compile model
    model.compile(optimizer=DEFAULT_OPT, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------- Plot diagnostic curves
def plot_diagnostics(history):
    # plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
        
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
        
	# save plot to file
	plt.savefig('./files/plots/' + str(TIMESTAMP) + '_model_' + MODEL_NO + '_plot.png')
	plt.close()

# ----------- Train and test model
def run_model():
    model = define_model(MODEL_NO)

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_datagen = test_datagen
    if MODEL_NO == '3-3' or MODEL_NO == '3-4':
        train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    train_data = train_datagen.flow_from_directory('dataset/train/',
                class_mode='binary', batch_size=64, target_size=(200, 200))
    test_data = test_datagen.flow_from_directory('dataset/test/',
                class_mode='binary', batch_size=64, target_size=(200, 200))
    
    # fit the model
    history = model.fit(train_data,
            steps_per_epoch=len(train_data),
            validation_data=test_data, 
            validation_steps=len(test_data), 
            epochs=EPOCHS, 
            verbose=1)

    # evaluate model
    _, acc = model.evaluate(test_data, steps=len(test_data), verbose=0)

    # plot curves
    plot_diagnostics(history)

    # save model
    model.save('files/models/' + str(TIMESTAMP) + '_model_' + MODEL_NO + '.h5')
    return (acc * 100.0)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# create file to save diagnostics
file_name_result = 'files/results/' + str(TIMESTAMP) + '_model_' + MODEL_NO + ".csv"

with open(file_name_result, 'a+') as file:
    file.write('iteration;datetime;elapsed_time;acc' + '\n')

    # execute training runs
    for i in range(RUNS):
        #start timer
        time_begin = time.time()

        #create new env for training
        prepare_datasets()

        #run model
        acc = run_model()

        #stop timer
        time_end = time.time()
        time_elapsed = time_end - time_begin
        print('Accuracy: ' + str(acc))
        print('Total time for processing model_' + MODEL_NO + ': ' + str(time_elapsed))
        file.write(str(i + 1) + ';' + str(TIMESTAMP) + ';' + str(time_elapsed) + ';' + str(acc) + '\n')