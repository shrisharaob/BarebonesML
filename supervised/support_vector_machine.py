# Support Vector Machine: Linear
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# cwd
directory = Path(__file__).absolute()
# setting path
sys.path.append(str(directory.parent.parent))
from utils.train_test_split import train_test_split
from utils.generate_toy_data import generate_data, generate_scaled_data


class SVM:

    def __init__(self,
                 learning_rate=0.01,
                 lambda_param=0.01,
                 num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _compute_cost(self, X, y):
        """The cost function is used here is known as the hinge loss. It
        measures the classification error.  The lambda_param is multiplied with
        the hinge loss term. This essentially adds a regularization penalty to
        the cost function.  By adjusting the value of lambda_param, you can
        control the impact of regularization on the overall cost
        function. Higher values of lambda_param increase the regularization
        strength, penalizing complex models more heavily.Adjusting its value
        allows you to fine-tune the balance between maximizing the margin and
        minimizing the classification error, thus influencing the model's
        generalization capability.
        """
        m = X.shape[0]
        distances = 1 - y * (np.dot(X, self.weights) - self.bias)
        distances[distances < 0] = 0  # max(0, distance)
        hinge_loss = self.lambda_param * (np.sum(distances) / m)
        cost = 1 / 2 * np.dot(self.weights, self.weights) + hinge_loss
        return cost

    def _compute_gradients(self, X, y):
        m = X.shape[0]
        distances = 1 - y * (np.dot(X, self.weights) - self.bias)
        dw = np.zeros_like(self.weights)
        db = 0

        for idx, distance in enumerate(distances):
            if max(0, distance) == 0:
                di = self.weights
                dw += di
                db += 0
            else:
                di = self.weights - self.lambda_param * y[idx] * X[idx]
                dw += di
                db += -self.lambda_param * y[idx]

        dw /= m
        db /= m
        return dw, db

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.num_iterations):
            dw, db = self._compute_gradients(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)


def grid_search(X_train, y_train, X_val, y_val, learning_rates, lambda_params,
                num_iterations_list):
    """ds"""
    best_accuracy = 0
    best_hyperparameters = None

    learning_rate_accuracies = []
    lambda_accuracies = []

    for learning_rate in learning_rates:
        lambda_accuracies_row = []
        for lambda_param in lambda_params:
            model = SVM(learning_rate=learning_rate,
                        lambda_param=lambda_param,
                        num_iterations=num_iterations_list[-1])
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == y_val)
            lambda_accuracies_row.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = {
                    'learning_rate': learning_rate,
                    'lambda': lambda_param,
                    'num_iterations': num_iterations_list[-1]
                }

        learning_rate_accuracies.append(lambda_accuracies_row)

    # Plot Learning Rate vs. Accuracy
    plt.figure(figsize=(10, 6))
    for i, lambda_param in enumerate(lambda_params):
        plt.plot(learning_rates, [row[i] for row in learning_rate_accuracies],
                 'o-',
                 label=f'Lambda={lambda_param:.4f}')

    plt.title('Learning Rate vs. Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Lambda vs. Accuracy
    plt.figure(figsize=(10, 6))
    for i, learning_rate in enumerate(learning_rates):
        plt.plot(lambda_params,
                 learning_rate_accuracies[i],
                 'o-',
                 label=f'Learning Rate={learning_rate:.4f}')

    plt.title('Lambda vs. Accuracy')
    plt.xlabel('Lambda Parameter')
    plt.ylabel('Accuracy')
    plt.legend()
    return best_hyperparameters, best_accuracy


# def generate_data():
#     np.random.seed(42)
#     X = np.random.randn(100, 2)
#     y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
#     return X, y

# def generate_scaled_data():
#     np.random.seed(42)

#     # Generate synthetic data with correlated features
#     mean = [0, 0]
#     cov = [[1, 0.8], [0.8, 1]]
#     X = np.random.multivariate_normal(mean, cov, 100)

#     # Assign labels based on the sum of features
#     y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

#     # Scale the features manually
#     mean_X = np.mean(X, axis=0)
#     std_X = np.std(X, axis=0)
#     X_scaled = (X - mean_X) / std_X

#     return X_scaled, y


def plot_decision_boundary(X, y, svm):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0],
                X[:, 1],
                c=y,
                cmap=plt.cm.Paired,
                edgecolors='k',
                marker='o',
                s=100)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx,
                yy,
                Z,
                levels=[-1, 0, 1],
                linestyles=['--', '-', '--'],
                colors='k',
                alpha=0.7)

    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# def train_val_split(X, y, val_ratio=0.2, random_seed=None):
#     """
#     Split the dataset into training and validation sets.

#     Parameters:
#     - X: Input features
#     - y: Labels
#     - val_ratio: Ratio of the validation set size
#     - random_seed: Seed for reproducibility

#     Returns:
#     - X_train, y_train, X_val, y_val
#     """
#     if random_seed is not None:
#         np.random.seed(random_seed)

#     m = X.shape[0]
#     indices = np.arange(m)
#     np.random.shuffle(indices)

#     val_size = int(m * val_ratio)
#     val_indices = indices[:val_size]
#     train_indices = indices[val_size:]

#     X_train, y_train = X[train_indices], y[train_indices]
#     X_val, y_val = X[val_indices], y[val_indices]

#     return X_train, y_train, X_val, y_val


def main():
    # Generate synthetic data
    X, y = generate_scaled_data()

    # Convert labels to -1 and 1
    y[y == 0] = -1

    # Fit SVM model
    svm = SVM(learning_rate=0.01, lambda_param=0.01, num_iterations=1000)
    svm.fit(X, y)

    # Plot decision boundary
    plot_decision_boundary(X, y, svm)


def main2():
    # Define the hyperparameter search space
    learning_rates = np.linspace(1e-3, 9e-1, 10)  #[0.001, 0.01, 0.1]
    lambda_params = np.linspace(1e-3, 5e-1, 3)  # [0.001, 0.01, 0.1]
    num_iterations_list = [100, 1000]

    X, y = generate_scaled_data()

    X_train, y_train, X_val, y_val = train_test_split(X,
                                                      y,
                                                      val_ratio=0.2,
                                                      random_seed=42)

    best_hyperparameters, best_accuracy = grid_search(X_train, y_train, X_val,
                                                      y_val, learning_rates,
                                                      lambda_params,
                                                      num_iterations_list)

    print(best_hyperparameters)

    # Fit SVM model
    svm = SVM(learning_rate=best_hyperparameters['learning_rate'],
              lambda_param=best_hyperparameters['lambda'],
              num_iterations=1000)
    svm.fit(X, y)

    # Plot decision boundary
    plot_decision_boundary(X, y, svm)


if __name__ == "__main__":
    main()
