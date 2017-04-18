# The-Nature-Conservancy-Fisheries-Monitoring
Kaggle Competition

## Files
* install-requirements.sh -- Install necessary packages and dependencies
* inception_ft.py		-- Main InceptionV3 CNN implementation file
* lgb.py			-- GBDT implementation using [Microsoft LightGBM](https://github.com/Microsoft/LightGBM)

## Compute environment setup
__NOTE__: This experiment will not run in a normal laptop/PC or will take a long time.
We suggest a system with a powerful GPU for tensor processing.

Our setup: ARC compute node **c74/c76** with Nvidia GTX TitanX GPU, 3072 cores, 1000 MHz core clock, 12 GB memory

Setup.sh script provides code for getting most required packages set up on an ARC node.

## How to run
	python inception_ft.py train	# Train the InceptionV3 model
	python inception_ft.py predict	# Generate embeddings for train and test images
	python lgb.py

## Details
![Image](https://github.com/abhinavchdhry/The-Nature-Conservancy-Fisheries-Monitoring/blob/master/Model.png)

We use a Keras InceptionV3 model with weights trained on Imagenet.
On top, we add the following layers to generate a 8-class probability vector corresponding to 8 classes in our dataset.

	Dropout(0.5)
	GlobalAveragePooling2D()
	Dense(1024, 'relu')	--- 1
	Dense(8, 'softmax')	--- 2
	
First only the added top layers are trained by making InceptionV3 layers untrainable.
Next, the InceptionV3 model is finetuned to our dataset by training only layers with indices 171 onwards. This essentially keeps the top half of the Inception model as it is.
Now, remove the added top Dense layers (1 and 2)
Output of CNN is now a 2048-vector. This will let us generate embeddings for our images.

Generate embeddings of training and test dataset and output to CSV files.

### LightGBM for fast gradient boosting trees
Use LightGBM to fit gradient boosted trees on the training set embeddings.
Use the best cross validated model for test set probabilities.

1. model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_leaves=60, max_depth=5, learning_rate=0.01, n_estimators=200, subsample=1, colsample_bytree=0.8, reg_lambda=0)

## Results
Public leaderboard log-loss: **1.28766** with above settings
All models with higher n_estimators seem to overfit resulting in worse log-loss scores
