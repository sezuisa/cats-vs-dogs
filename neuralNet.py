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
#from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers


import matplotlib.image as mpimg
from datetime import datetime
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

base_dir = 'data'
model_no = '3-4'
seedNum = 1
default_opt = optimizers.Adam(learning_rate=0.001)

# ----------- Prepare datasets

# train_data = image_dataset_from_directory(base_dir,
#                                                 image_size=(200,200),
#                                                 subset='training',
#                                                 seed = seedNum,
#                                                 label_mode='binary',
#                                                 validation_split=0.2,
#                                                 batch_size= 64)
# test_data = image_dataset_from_directory(base_dir,
#                                                 image_size=(200,200),
#                                                 subset='validation',
#                                                 seed = seedNum,
#                                                 label_mode='binary',
#                                                 validation_split=0.2,
#                                                 batch_size= 64)




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
seed(seedNum)
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

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_datagen = test_datagen
if model_no == '3-3' or model_no == '3-4':
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_data = train_datagen.flow_from_directory('dataset/train/',
            class_mode='binary', batch_size=64, target_size=(200, 200))
test_data = test_datagen.flow_from_directory('dataset/test/',
            class_mode='binary', batch_size=64, target_size=(200, 200))



# ----------- Define the model
def define_model(model_no):
    model = Sequential()
    # Model 1: VGG-1
    if model_no == '1' or model_no == '1-2':
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
    model.compile(optimizer=default_opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------- Train and test model

startTime = datetime.now()
tf.random.set_seed(seedNum)

model = define_model(model_no)

model.summary()

# visualise model
keras.utils.plot_model(
    model,
    to_file='files/model_' + model_no + '.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)

# fit the model
history = model.fit(train_data,
        steps_per_epoch=len(train_data),
        validation_data=test_data, 
        validation_steps=len(test_data), 
        epochs=100, 
        verbose=1)

# evaluate model
_, acc = model.evaluate(test_data, steps=len(test_data), verbose=0)
print('> %.3f' % (acc * 100.0))

# save model
model.save("models/model_" + model_no + ".h5", overwrite=True)

print('Total time for processing model_' + model_no + ':', (datetime.now() - startTime))

# save accuracy and loss diagrams
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
figs[0].savefig("files/model_" + model_no + "_loss.png")
figs[1].savefig("files/model_" + model_no + "_accuracy.png")