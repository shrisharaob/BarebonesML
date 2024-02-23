import numpy as np
import matplotlib.pyplot as plt


class KNN(object):
    """KNN(k)
    """

    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, X):
        # Compute Euclidean distance between the input X and each training sample
        return np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)

    def _k_nearest(self, distances):
        # Find indices of k nearest neighbors for each input sample
        sorted_dists = np.argsort(distances, axis=1)  # ascending
        k_nearest_idxs = sorted_dists[:, :self.k]
        return k_nearest_idxs

    def _get_label(self, X):
        # Get labels of k nearest neighbors and compute the mean
        distances = self._euclidean_distance(X)
        k_nearest_idxs = self._k_nearest(distances)
        k_nearest_labels = self.y_train[k_nearest_idxs]
        return np.mean(k_nearest_labels, axis=1)

    def predict(self, X):
        # Predict labels for input X
        return self._get_label(X.astype(float))


def generate_data(num_samples=300, num_classes=3, num_features=2):
    np.random.seed(2)
    mean1 = [0, 0]
    cov1 = [[1, 0.5], [0.5, 1]]
    mean2 = [2, 2]
    cov2 = [[1, -0.5], [-0.5, 1]]
    mean3 = [-2, 2]
    cov3 = [[1, 0], [0, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, num_samples // 3)
    X2 = np.random.multivariate_normal(mean2, cov2, num_samples // 3)
    X3 = np.random.multivariate_normal(mean3, cov3, num_samples // 3)

    # Generate random means and covariance matrices for each class
    # means = np.random.uniform(low=-2, high=2, size=(num_classes, num_features))
    # covs = np.empty((num_classes, num_features, num_features))
    # for i in range(num_classes):
    #     covs[i] = np.random.uniform(low=0,
    #                                 high=1,
    #                                 size=(num_features, num_features))
    #     covs[i] = np.dot(
    #         covs[i],
    #         covs[i].T)  # Ensure covariance matrix is positive semi-definite

    # X1 = np.random.multivariate_normal(means[0], covs[0],
    #                                    num_samples // num_classes)
    # X2 = np.random.multivariate_normal(means[1], covs[1],
    #                                    num_samples // num_classes)
    # X3 = np.random.multivariate_normal(means[2], covs[2],
    #                                    num_samples // num_classes)

    X = np.concatenate([X1, X2, X3])
    y = np.concatenate([
        np.zeros(num_samples // 3),
        np.ones(num_samples // 3),
        np.ones(num_samples // 3) * 2
    ])

    return X, y


# def generate_data(num_samples=300,
#                   num_features=2,
#                   num_classes=2,
#                   random_state=42):
#     np.random.seed(random_state)
#     X = np.random.randn(num_samples, num_features)
#     y = np.random.randint(num_classes, size=num_samples)
#     return X, y

# def visualize_data(X, y):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X[:, 0],
#                 X[:, 1],
#                 c=y,
#                 cmap=plt.cm.bwr,
#                 marker='o',
#                 edgecolors='k')
#     plt.title('Generated Data')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.colorbar(label='Class')
#     plt.grid(True)
#     plt.show()


def demo(k=3, ax=None):
    # Generate synthetic data
    X, y = generate_data()

    # # Visualize the data
    # visualize_data(X, y)

    # Split the data into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Initialize and train the KNN classifier
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    # Predict labels for the test data
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy of KNN classifier with k={k}: {accuracy:.2f}')

    # Visualize the decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plt.figure(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
    ax.scatter(X[:, 0],
               X[:, 1],
               c=y,
               cmap=plt.cm.bwr,
               marker='o',
               edgecolors='k')
    ax.set_title(f'k:{k}  acc:{accuracy:.2f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    # plt.colorbar(label='Class')
    plt.grid(True)


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    demo(k=2, ax=ax[0])
    demo(k=5, ax=ax[1])
    demo(k=13, ax=ax[2])
    fig.suptitle('KNN Decision Boundries')

    plt.show()
