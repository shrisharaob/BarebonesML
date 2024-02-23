import numpy as np
import matplotlib.pyplot as plt


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Compute mean of the data
        self.mean = np.mean(X, axis=0)
        # Center the data
        centered_data = X - self.mean
        # Compute covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)
        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Sort eigenvectors based on eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        # Select top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data
        centered_data = X - self.mean
        # Project the data onto the principal components
        transformed_data = np.dot(centered_data, self.components)
        return transformed_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def demo_pca():
    # Generate some random data
    np.random.seed(0)
    X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

    # Apply PCA
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(X)

    # Plot original data
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
    plt.title('Original Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.ylim(-2, 2)

    # Plot transformed data
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.8)
    plt.title('Transformed Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.ylim(-2, 2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo_pca()
