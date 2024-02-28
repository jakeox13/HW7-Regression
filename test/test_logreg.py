"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import sklearn
import numpy as np
from regression import (logreg, utils)
# (you will probably need to import more things here)

def test_prediction():
	pass

def test_loss_function():
	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = sklearn.preprocessing.StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	log_model.train_model(X_train, y_train, X_val, y_val)
	y_pred=log_model.make_prediction(X_train)
	assert sklearn.metrics.log_loss(y_val,y_pred) == log_model.loss_function(y_val,y_pred)

def test_gradient():
	pass

def test_training():
	
	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = sklearn.preprocessing.StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	# Save intial weights
	inital_w=log_model.W

	# train model
	log_model.train_model(X_train, y_train, X_val, y_val)

	assert inital_w != log_model.W