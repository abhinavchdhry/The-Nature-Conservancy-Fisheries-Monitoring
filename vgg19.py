import os, sys, glob

from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import numpy as np
import pandas as pd
import datetime
import os, sys, glob

import theano

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
from keras.utils import to_categorical
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.utils import get_file

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./TRAIN/"
VALIDATION_PATH = "./VALID/"

def loadTrainAndValidationDatasets(size):
	print("### Creating generators...")
	train_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()

	#TRAIN_PATH = "./sample/TRAIN/"
	#VALIDATION_PATH = "./sample/VALID/"
	classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
	train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=size, batch_size=1, class_mode='sparse', classes=classes)
	valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=size, batch_size=1, class_mode='sparse', classes=classes)

	#TEST_PATH = "./TEST/"
	#test_datagen = ImageDataGenerator()
	#test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)

	print("### Loading datasets...")
	train_len = len(glob.glob('./TRAIN/*/*.jpg'))
	valid_len = len(glob.glob('./VALID/*/*.jpg'))

	xytuples = []
	for i in range(train_len):
	        x = train_generator.next()
	        xytuples.append(x)

	train_X = np.concatenate([x[0] for x in xytuples])
	train_Y = np.concatenate([y[1] for y in xytuples])

	xytuples = []
	for i in range(valid_len):
	        x = valid_generator.next()
	        xytuples.append(x)

	valid_X = np.concatenate([x[0] for x in xytuples])
	valid_Y = np.concatenate([y[1] for y in xytuples])

	print("Train X shape = " + str(train_X.shape))
	print("Train Y shape = " + str(train_Y.shape))
	train_X = train_X / 255
	valid_X = valid_X / 255

	return(train_X, train_Y, valid_X, valid_Y)


def createVGG19(input_shape=(512, 512, 3), optimizer=Adam(lr=0.001), weights = 'imagenet'):
	base_model = VGG19(weights=weights, include_top=False, input_shape=input_shape,pooling='avg')
	
	x = base_model.layers[-1].output
	x = Dropout(0.3)(x)
#	x = GlobalAveragePooling2D()(x)
#	x = Dense(512, activation='relu')(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.3)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dense(8, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=x)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy")

	return(base_model, model)


size = (512, 512)

train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets(size)
train_Y = to_categorical(train_lbls)
valid_Y = to_categorical(valid_lbls)

optimizer = SGD(lr=1e-3, decay=1e-4, momentum=0.89, nesterov=False)
base,model = createVGG19(optimizer = optimizer)

for layer in base.layers:
		layer.trainable = False

print("### Compiling model...")
	
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

print("### Fitting model...")
model.fit(x=train_X, y=train_Y, batch_size=32, epochs=7, verbose=2, validation_data=(valid_X, valid_Y) )


print("### Finetune phase started...")

for layer in base.layers[16:]:
		layer.trainable = True

model.compile(optimizer=optimizer,  loss="categorical_crossentropy", metrics=['accuracy'])

print("### Fitting model...")
model.fit(x=train_X, y=train_Y, batch_size=32, epochs=7, verbose=2, validation_data=(valid_X, valid_Y) )

print("### Saving weights...")
model.save_weights("vgg_19_weights.h5")
