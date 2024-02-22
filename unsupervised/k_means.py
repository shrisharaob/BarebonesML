import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """KMeans(n_clusters=2, max_iter=300, tol=1e-4)
    """

    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize cluster centers randomly
        random_indices = np.random.choice(n_samples,
                                          self.n_clusters,
                                          replace=False)
        self.cluster_centers_ = X[random_indices]

        for _ in range(self.max_iter):
            # Assign each sample to the nearest cluster
            distances = np.sqrt(
                ((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update cluster centers
            new_centers = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if self._has_converged(new_centers):
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels

    def predict(self, X):
        distances = np.sqrt(
            ((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _has_converged(self, new_centers):
        # Check if the distance between new and old cluster centers is less than the tolerance
        return np.linalg.norm(new_centers - self.cluster_centers_) < self.tol


def visualize_kmeans(X, kmeans):
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X[:, 0],
                X[:, 1],
                c=kmeans.labels_,
                cmap='viridis',
                s=50,
                alpha=0.5)

    # Plot cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                c='red',
                marker='x',
                s=200,
                label='Centroids')

    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def demo_kmeans():
    # Generate synthetic data
    np.random.seed(0)
    X = np.concatenate([
        np.random.randn(100, 2) * 2 + [2, 2],
        np.random.randn(100, 2) * 1 + [-2, -2],
        np.random.randn(100, 2) * 0.5 + [2, -2]
    ])

    # Initialize and fit KMeans model
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    # Visualize clusters
    visualize_kmeans(X, kmeans)


if __name__ == '__main__':
    demo_kmeans()
