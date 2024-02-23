import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np


class GaussianDiscriminantAnalysis:

    def __init__(self):
        self.mu = None
        self.sigma = None
        self.phi = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.mu = []
        self.sigma = []
        self.phi = []

        for c in range(self.n_classes):
            X_c = X[y == c]
            mu_c = np.mean(X_c, axis=0)
            sigma_c = np.cov(X_c.T)
            phi_c = len(X_c) / len(X)

            self.mu.append(mu_c)
            self.sigma.append(sigma_c)
            self.phi.append(phi_c)

    def predict(self, X):
        likelihoods = np.zeros((X.shape[0], self.n_classes))

        for c in range(self.n_classes):
            likelihoods[:, c] = self._gaussian_pdf(X, self.mu[c],
                                                   self.sigma[c])

        posterior = likelihoods * self.phi
        predictions = np.argmax(posterior, axis=1)
        return predictions

    def _gaussian_pdf(self, X, mu, sigma):
        n = X.shape[1]
        det_sigma = np.linalg.det(sigma)
        constant = 1 / ((2 * np.pi)**(n / 2) * det_sigma**0.5)
        exponent = -0.5 * np.sum(
            np.dot(X - mu, np.linalg.inv(sigma)) * (X - mu), axis=1)
        return constant * np.exp(exponent)


def generate_data(num_samples=300, num_classes=3, num_features=2):
    # np.random.seed(2)
    # mean1 = [0, 0]
    # cov1 = [[1, 0.5], [0.5, 1]]
    # mean2 = [2, 2]
    # cov2 = [[1, -0.5], [-0.5, 1]]
    # mean3 = [-2, 2]
    # cov3 = [[1, 0], [0, 1]]
    # X1 = np.random.multivariate_normal(mean1, cov1, num_samples // 3)
    # X2 = np.random.multivariate_normal(mean2, cov2, num_samples // 3)
    # X3 = np.random.multivariate_normal(mean3, cov3, num_samples // 3)

    # Generate random means and covariance matrices for each class
    means = np.random.uniform(low=-2, high=2, size=(num_classes, num_features))
    covs = np.empty((num_classes, num_features, num_features))
    for i in range(num_classes):
        covs[i] = np.random.uniform(low=0,
                                    high=1,
                                    size=(num_features, num_features))
        covs[i] = np.dot(
            covs[i],
            covs[i].T)  # Ensure covariance matrix is positive semi-definite

    X1 = np.random.multivariate_normal(means[0], covs[0],
                                       num_samples // num_classes)
    X2 = np.random.multivariate_normal(means[1], covs[1],
                                       num_samples // num_classes)
    X3 = np.random.multivariate_normal(means[2], covs[2],
                                       num_samples // num_classes)

    X = np.concatenate([X1, X2, X3])
    y = np.concatenate([
        np.zeros(num_samples // 3),
        np.ones(num_samples // 3),
        np.ones(num_samples // 3) * 2
    ])

    return X, y


def visualize_decision_boundary(X, y, classifier):
    h = 0.02  # step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot the decision boundaries
    Z = Z.astype(np.int64)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Gaussian Discriminant Analysis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def demo():
    X, y = generate_data()
    gda = GaussianDiscriminantAnalysis()
    gda.fit(X, y)
    visualize_decision_boundary(X, y, gda)


if __name__ == "__main__":
    demo()
