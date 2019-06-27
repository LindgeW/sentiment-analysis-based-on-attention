from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


if __name__ == '__main__':
	iris = load_iris()
	X = iris.data  # (150, 4)
	y = iris.target  # (150, )  取值0，1，2
	# print(X.shape, y.shape)
	# print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
	print(X[:2])
	pca_reduced_x = PCA(n_components=2).fit_transform(X)
	tsne_reduced_x = TSNE(n_components=2, init='pca', learning_rate=100).fit_transform(X)

	plt.figure(figsize=(12, 6))
	plt.subplot(121)
	plt.scatter(pca_reduced_x[:, 0], pca_reduced_x[:, 1], c=y, label='PCA')
	plt.subplot(122)
	plt.scatter(tsne_reduced_x[:, 0], tsne_reduced_x[:, 1], c=y, label='TSNE')
	plt.legend()
	plt.colorbar()
	plt.show()
