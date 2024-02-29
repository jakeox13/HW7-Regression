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
	'''
	Function for testing the correct implemention of the loss function for logistic regressor class
	'''
	# Initlize model
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	
	# If prediction is working correctly then a vector of 0 as input should lead to a vector containing all 0.5s as output
	y_pred=log_model.make_prediction(np.zeros(len(log_model.W)))

	assert np.array_equal(y_pred,np.array([0.5]*len(log_model.W)))


def test_loss_function():
	'''
	Function for testing the correct implemention of the loss function for logistic regressor class
	'''
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
	
	#Intilze the model
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	
	log_model.train_model(X_train, y_train, X_val, y_val)

	# Padding data with vector of ones for bias term
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	y_pred=log_model.make_prediction(X_val)
	assert sklearn.metrics.log_loss(y_val,y_pred) == log_model.loss_function(y_val,y_pred)

def test_gradient():
	'''
	Function for testing the correct implemention of gradient calculation for logistic regressor class
	'''
	# Example input data
	X = np.array([[1, 2], [2, 3], [3, 4]])
	y_true = np.array([0, 1, 0])  # Binary labels
	weights = np.array([0.5, -0.5])  # Example weights

	# Set up toy model
	log_model = logreg.LogisticRegressor(num_feats=1, learning_rate=0.00001, tol=0.01, max_iter=1, batch_size=1)
	log_model.W=weights
	
	# Manually compute gradients for comparison
	

	test=log_model.calculate_gradient(y_true,X)
	assert (np.allclose(test,np.array([[-0.26524401, -0.39786602]]), rtol=1e-05))

def test_training():
	'''
	Function for testing the correct implemention of training for logistic regressor class
	'''
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


	# Check if any element is different Weights should be different from start
	assert np.array_equal(inital_w, log_model.W)==False

test_loss_function()
test_training()
test_prediction()
test_gradient()