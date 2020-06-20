import numpy as np
def accuracy_score(y_pred, y_test):
	assert y_pred.shape == y_test.shape, "维度必须相同"
	return np.sum(y_pred == y_test) / len(y_test)

def r2_score(y_pred, y_test):
	assert y_pred.shape == y_test.shape, "维度必须相同"
	return 1 - mean_squared_error(y_pred, y_test) / np.var(y_test)

def mean_squared_error(y_pred, y_test):
	assert y_pred.shape == y_test.shape, "维度必须相同"
	return np.sum((y_pred - y_test)**2) / len(y_test)

def mean_absolute_error(y_pred, y_test):
	assert y_pred.shape == y_test.shape, "维度必须相同"
	return np.sum(np.absolute(y_pred - y_test)) / len(y_test)


