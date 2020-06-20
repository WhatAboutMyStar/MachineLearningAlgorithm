import numpy as np
from .metrics import accuracy_score
class KNeighborsClassifier:
	def __init__(self, k=3):
		assert k > 0, "k必须大于0"
		self.k = k
		self.x_train = None
		self.y_train = None

	def fit(self, x_train, y_train):
		assert x_train.shape[0] == y_train.shape[0], "训练集数量和标签数量必须相同"
		assert x_train.ndim == 2
		self.x_train = x_train
		self.y_train = y_train
		return self

	def predict(self, x_predict):
		assert self.x_train is not None and self.y_train is not	None , "必须先fit"
		assert x_predict.ndim == 2
		y_predict = [self._predict(x) for x in x_predict]
		return np.array(y_predict)

	def _predict(self, x):
		distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.x_train]
		nearest = np.argsort(distances)[:self.k]
		top_k_y = [self.y_train[index] for index in nearest]
		#下面这部分其实可以用Counter实现
		d = {}
		for cls in top_k_y:
			d[cls] = d.get(cls, 0) + 1
		d_list = list(d.items())
		d_list.sort(key=lambda x: x[1], reverse=True)
		return np.array(d_list[0][0])

	def score(self, x_test, y_test):
		assert x_test.ndim == 2
		y_pred = self.predict(x_test)
		return accuracy_score(y_pred, y_test)

	def __repr__(self):
		return "KNN(k={})".format(self.k)
