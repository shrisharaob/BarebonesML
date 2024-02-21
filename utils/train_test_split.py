import numpy as np


def train_test_split(X, y, val_ratio=0.2, random_seed=None):
    """
    Split the dataset into training and validation sets.

    Parameters:
    - X: Input features
    - y: Labels
    - val_ratio: Ratio of the validation set size
    - random_seed: Seed for reproducibility

    Returns:
    - X_train, y_train, X_val, y_val
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    val_size = int(m * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    return X_train, y_train, X_val, y_val
