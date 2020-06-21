import numpy as np
class KMeans:
	def __init__(self, n_cluster=3, random_state=666666):
		self.cluster_centers_ = None
		self.n_cluster = n_cluster
		self.labels_ = None
		if random_state:
			np.random.seed(random_state)

	def fit(self, x_train):
		assert x_train.ndim == 2

		choice = np.random.choice(len(x_train),
								  self.n_cluster,
								  replace=False)

		self.cluster_centers_ = x_train[choice]
		last_centroid = self.cluster_centers_ + 10

		# 质心没什么变化后就停止
		while np.sum(np.absolute(self.cluster_centers_ - last_centroid)) > 0.000001:
			last_centroid = self.cluster_centers_
			label = []
			for data in x_train:
				near = float('inf')
				flag = 0
				for i, ch in enumerate(self.cluster_centers_):
					dis = np.sqrt(np.sum((ch - data) ** 2))
					if dis < near:
						near = dis
						flag = i
				label.append(flag)

			for i in range(self.n_cluster):
				new_centroid = np.zeros(x_train.shape[1])
				cnt = 0
				for index, data in enumerate(x_train):
					if label[index] == i:
						cnt += 1
						new_centroid += data
				new_centroid /= cnt
				self.cluster_centers_[i] = new_centroid
		self.labels_ = np.array(label)
		return self

	def predict(self, x_test):
		assert x_test.ndim == 2

		label = []
		for data in x_test:
			near = float('inf')
			flag = 0
			for i, ch in enumerate(self.cluster_centers_):
				dis = np.sqrt(np.sum((ch - data) ** 2))
				if dis < near:
					near = dis
					flag = i
			label.append(flag)
		return np.array(label)

	def __repr__(self):
		return "KMeans(n_cluster={})".format(self.n_cluster)
