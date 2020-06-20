import numpy as np
from .metrics import accuracy_score
from .metrics import r2_score
class LinearRegression:
	def __init__(self):
		self._theta = None
		self.intercept_ = None
		self.coef_ = None

	def fit(self, x_train, y_train):
		assert x_train.ndim == 2
		assert len(x_train) == len(y_train)

		X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
		self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
		self.intercept_ = self._theta[0]
		self.coef_ = self._theta[1:]

		return self

	def predict(self, x_predict):
		assert self.coef_ is not None
		assert self.intercept_ is not None
		assert self._theta is not None
		assert x_predict.ndim == 2

		X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
		return X_b.dot(self._theta)

	def score(self, x_test, y_test):
		assert x_test.ndim == 2
		assert len(x_test) == len(y_test)

		y_pred = self.predict(x_test)
		return r2_score(y_pred, y_test)

	def __repr__(self):
		return "LinearRegression()"


class LogisticRegression:
	def __init__(self, learning_rate=0.001, max_iter=10000):
		self._theta = None
		self.intercept_ = None
		self.coef_ = None
		self.learning_rate = learning_rate
		self.max_iter = max_iter

	def _sigmoid(self, z):
		return 1. / (1. + np.exp(-z))

	def fit(self, x_train, y_train):
		def J(theta, X_b, y_train):
			y_hat = self._sigmoid(X_b.dot(theta))
			return - np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat)) / len(y_train)

		def dJ(theta, X_b, y_train):
			y_hat = self._sigmoid(X_b.dot(theta))
			return X_b.T.dot(y_hat - y_train) / len(y_train)

		assert len(x_train) == len(y_train)
		assert x_train.ndim == 2

		X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
		self._theta = np.random.randn(X_b.shape[1])  # 这里我用了随机初始化，初始化为正态分布
		iter_num = 0
		while iter_num < self.max_iter:
			iter_num += 1
			last_theta = self._theta
			self._theta = self._theta - self.learning_rate * dJ(self._theta, X_b, y_train)
			if (abs(J(self._theta, X_b, y_train) - J(last_theta, X_b, y_train)) < 1e-7):
				break

		self.intercept_ = self._theta[0]
		self.coef_ = self._theta[1:]
		return self

	def predict(self, x_predict):
		assert x_predict.ndim == 2
		assert self.intercept_ is not None
		assert self.coef_ is not None
		assert self._theta is not None

		X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
		y_predict = self._sigmoid(X_b.dot(self._theta))
		y_predict = np.array(y_predict >= 0.5, dtype='int')
		return y_predict

	def score(self, x_test, y_test):
		assert x_test.ndim == 2
		assert len(x_test) == len(y_test)

		y_predict = self.predict(x_test)
		return accuracy_score(y_predict, y_test)

	def __repr__(self):
		return "LogisticRegression()"

