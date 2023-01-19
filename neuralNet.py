import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import image_dataset_from_directory, load_img
from keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.image as mpimg


path = 'dogs-and-cats'
classes = os.listdir(path)
print(classes)