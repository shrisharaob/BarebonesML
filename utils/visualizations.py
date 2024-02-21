import numpy as np
import matplotlib.pyplot as plt


def plot_data_with_labels(X, y_test, y_pred):

    plt.figure(figsize=(8, 6))

    # Plotting points with true labels
    plt.scatter(X[:, 0],
                X[:, 1],
                c=y_test,
                cmap=plt.cm.Paired,
                marker='o',
                label='True Labels')

    # Plotting points with predicted labels
    plt.scatter(X[:, 0],
                X[:, 1],
                c=y_pred,
                marker='x',
                cmap=plt.cm.Paired,
                label='Predicted Labels')

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('True Labels vs. Predicted Labels')
    plt.legend()
    plt.show()
