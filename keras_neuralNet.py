# save the final model to file
import sys
import time
from datetime import datetime
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random


#---------------------------------
#CONSTANTS
EPOCHS = 10
RUNS = 5
NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SEED_NUM = 1
DEFAULT_OPT = SGD(learning_rate=0.001, momentum=0.9)

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


# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))

	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False

	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)

	# define new model
	model = Model(inputs=model.inputs, outputs=output)

	# compile model
	model.compile(optimizer=DEFAULT_OPT, loss='binary_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
def plot_diagnostics(history, model_number):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')

	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

	# save plot to file
	pyplot.savefig('./files_keras/plots/keras_'+ str(NOW) + '_No_' + str(model_number) + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_model(count_of_epochs, model_number):
	# define model
	model = define_model()
	model.save('./files_keras/models/keras_model_' + str(NOW)+ '_No_' + str(model_number) +'.h5')

	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)

	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]

	# prepare iterator
	train_it = datagen.flow_from_directory('dataset/train/',
				class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('dataset/test/',
				class_mode='binary', batch_size=64, target_size=(224, 224))

	# fit model
	history = model.fit(train_it, 
			steps_per_epoch=len(train_it),
			validation_data=test_it, 
			validation_steps=len(test_it), 
			epochs=count_of_epochs, 
			verbose=1)

	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))

	# learning curves
	plot_diagnostics(history, model_number)
	return (acc * 100.0)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# create file to save diagnostics
file_name_result = 'files_keras/results' + str(NOW) + '.csv'

with open(file_name_result, 'a+') as file:
	file.write('model_number;datetime;elapsed_time;acc' + '\n')

    # execute training runs
	for i in range(RUNS):
		#start timer
		time_begin = time.time()

		#create new env for training
		prepare_datasets()

		#run model
		acc = run_model(EPOCHS, i)

		#stop timer
		time_end = time.time()
		time_elapsed = time_end - time_begin
		file.write(str(i) + ';' + str(NOW) + ';' + str(time_elapsed) + ';' + str(acc) + '\n')
	


