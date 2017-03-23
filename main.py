## Abhinav Choudhury
## Kaggle: The Nature Conservancy Fisheries Monitoring challenge

import os, sys, glob
from sklearn.model_selection import StratifiedShuffleSplit
import shutil

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

# Perform a stratified split given a vector of images names
# and a vector of corresponding image classes
def stratifiedSplit(X, Y, test_size=0.3):
	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=2017)
	sss.get_n_splits(X, Y)
	for train_idx, test_idx in sss.split(X, Y):
		break
	return(train_idx, test_idx)

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./TRAIN/"
VALIDATION_PATH = "./VALID/"

def createTrainAndValidationDatasets(datapath):
	print("### Sampling training and validation datasets...")
	classes = ["LAG", "DOL", "OTHER", "BET", "ALB", "NoF", "YFT", "SHARK"]

	def makeDirectoryStructure(path):
		if not os.path.exists(path):
			for cl in classes:
				os.makedirs(path + cl)
		else:
			shutil.rmtree(path)
			for cl in classes:
				os.makedirs(path + cl)

	img_names = []
	img_classes = []
	for cl in classes:
		class_dir = ORIGINAL_DATAPATH + cl + "/"
		filepaths = glob.glob(class_dir + "*.jpg")
		for filepath in filepaths:
			img_names.append(os.path.basename(filepath))
			img_classes.append(cl)

	train_idx, valid_idx = stratifiedSplit(img_names, img_classes)
	makeDirectoryStructure(TRAIN_PATH)
	makeDirectoryStructure(VALIDATION_PATH)

	for idx in train_idx:
		img_name = img_names[idx]
		img_class = img_classes[idx]
		shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name, TRAIN_PATH + img_class + "/" + img_name)

	for idx in valid_idx:
		img_name = img_names[idx]
		img_class = img_classes[idx]
		shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name, VALIDATION_PATH + img_class + "/" + img_name)


createTrainAndValidationDatasets(ORIGINAL_DATAPATH)

# Define, create and return a Keras model
def createModel(input_shape=(512, 512, 3)):
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation='softmax'))

	return(model)

print("### Creating model...")
model = createModel((512, 512, 3))

print("### Compiling model...")
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])

print("### Creating generators...")
train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(512, 512), batch_size=32)
valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=(512, 512), batch_size=32)

print("### Fitting model...")
model.fit_generator(train_generator, steps_per_epoch=1, epochs=10, validation_data=valid_generator, validation_steps=20)
