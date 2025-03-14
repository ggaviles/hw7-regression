"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""
import math

# Imports
import pytest
import sklearn.linear_model

import regression.utils
import regression.logreg
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def test_prediction():
    X, y = regression.utils.loadDataset()  # Read in dataset X and labels y
    X_train, X_test, y_train, y_test = regression.utils.loadDataset(split_seed=16, split_percent=0.8)

    # Chose l2 penalty because also used L2 regularization in my implemented Logistic Regressor
    # Chose C to be 2 because I set my reg_param to be 0.5 and C is the inverse of the regularization parameter
    sklearn_model = LogisticRegression(penalty='l2', C=2)
    sklearn_model.fit(X_train, y_train)
    sklearn_pred_labels = sklearn_model.predict(X_test)

    # Initialize implemented regularized logistic regression model
    model = regression.logreg.LogisticRegressor(num_feats=X.shape[1], learning_rate=0.001, max_iter=10000,
                                                batch_size=20, reg_param=0.5)

    model.train_model(X_train, y_train, X_test, y_test)
    pred_labels = model.make_prediction(X_test)

    # Calculate the mean error between the matrix of labels generated by sklearn and the matrix of labels generated
    # by my LogisticRegressor
    error = np.mean(sklearn_pred_labels != pred_labels)

    # Check that the mean error between sklearn label prediction and my LogisticRegressor prediction is less than 50%
    assert error * 100 < 50

def test_loss_function():
    # Test dataset
    x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # sklearn implementation of binary cross entropy loss function
    logr = LogisticRegression(solver='lbfgs')
    logr.fit(x.reshape(-1, 1), y)

    y_pred = logr.predict_proba(x.reshape(-1, 1))[:, 1].ravel()
    sklearn_loss = log_loss(y, y_pred)

    # LogisticRegressor implementation of binary cross entropy loss function (without regularization)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

    # Calculate the separable parts of the loss function
    y_zero_loss = y * np.log(y_pred)
    y_one_loss = (1 - y) * np.log(1 - y_pred)

    model_loss = -np.mean(y_zero_loss + y_one_loss)

    assert sklearn_loss == model_loss


def test_gradient():
    # Test dataset
    x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # sklearn implementation of the gradient of the binary cross entropy loss function
    logr = LogisticRegression(solver='lbfgs')
    logr.fit(x.reshape(-1, 1), y)

    y_pred = logr.predict_proba(x.reshape(-1, 1))[:, 1].ravel()

    # Another implementation of the gradient of the binary cross entropy loss function
    logr_grad_val = []
    for i in range(len(y)):
        logr_grad_val.append(x[i] * (y_pred[i] - y[i]))
    logr_grad = -np.mean(logr_grad_val)

    # Logistic Regressor implementation of the gradient of the binary cross entropy loss function (w/o regularization)
    error = y_pred - y
    grad = -x.T.dot(error) / len(y)

    assert math.isclose(grad, logr_grad, abs_tol=10**-16)

def test_training():
    # Check that my weights update during training
    X, y = regression.utils.loadDataset()
    X_train, X_test, y_train, y_test = regression.utils.loadDataset(split_seed=16, split_percent=0.8)

    model = regression.logreg.LogisticRegressor(num_feats=X.shape[1], learning_rate=0.001, max_iter=10000,
                                                batch_size=10, reg_param=0.25)
    prev_weights = model.W
    model.train_model(X_train, y_train, X_test, y_test)
    weights_after_training = model.W

    # Check to see each weight changes after training.
    assert (prev_weights != weights_after_training).any()
