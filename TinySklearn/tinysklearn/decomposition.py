import numpy as np
class PCA:
	def __init__(self, n_components):
		self.explained_variance_ratio_ = None
		self.n_components = n_components
		self.top_n_vector = None
		self.mean_vector = None

	def fit(self, X_train):
		self.mean_vector = X_train.mean(axis=0)

		cov_matrix = np.cov(X_train.T)

		eig_value, eig_vector = np.linalg.eig(cov_matrix)
		self.explained_variance_ratio_ = eig_value / eig_value.sum()

		self.top_n_vector = eig_vector[:, :self.n_components].T

	def transform(self, X):
		return (X - self.mean_vector).dot(self.top_n_vector.T)

	def __repr__(self):
		return "PCA(n_components={})".format(self.n_components)