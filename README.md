# The-Nature-Conservancy-Fisheries-Monitoring
Kaggle Competition

## Experiment updates
Keras InceptionV3 model with the following top:

	Dropout(0.5)
	GlobalAveragePooling2D()
	Dense(1024, 'relu')	--- 1
	Dense(8, 'softmax')	--- 2
	
First trained only top by making InceptionV3 layers untrainable.

Next, finetune:
Make layers 171 onwards trainable, and train again.

Remove top Dense layers ( 1 and 2)
Output of CNN is now a 2048-vector

Generate embeddings of training and test dataset and output to file.

### LightGBM for fast gradient boosting machine
Use LightGBM for further optimization.

### Experiments:
1. model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_leaves=60, max_depth=5, learning_rate=0.01, n_estimators=200, subsample=1, colsample_bytree=0.8, reg_lambda=0)

Public leaderboard log-loss: **1.28766**
All models with highes n_estimator seem to overfit with worse log-loss scores
