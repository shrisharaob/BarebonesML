import numpy as np


def generate_data(num_samples=100, seed=42):
    np.random.seed(seed)
    X = np.random.randn(num_samples, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    return X, y


def generate_scaled_data(num_samples=100):
    np.random.seed(42)

    # Generate synthetic data with correlated features
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]
    X = np.random.multivariate_normal(mean, cov, num_samples)

    # Assign labels based on the sum of features
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

    # Scale the features manually
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_scaled = (X - mean_X) / std_X

    return X_scaled, y
