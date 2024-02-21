"""This module provides a basic implementation of linear regression, including
   three algorithms:

    - Closed-form solution: Utilizes the closed-form solution to directly
      compute model coefficients.
    - Least Mean Squares (LMS): Implements a gradient descent approach to
      optimize model parameters.
    - Locally Weighted Regression (LWR): Performs regression with locally
      weighted instances.

    The module assumes the target variable y given the input features X and
    parameters θ follows a normal distribution N(μ, σ^2), where μ is the mean
    and σ^2 is the variance.
"""

import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt  # noqa
import os


class LinearRegression:
    """A simple Python class for Linear Regression using NumPy.

    Attributes:
        theta (numpy.ndarray): Coefficients of the linear regression model,
        including the bias term.

    Methods:
        fit(X, y, method='closed_form'): Fit the linear regression model to the
        training data.
        predict(X): Make predictions using the trained model.
    """

    def __init__(self):
        """Initialize LinearRegression object."""
        self.theta = None

    def _add_bias(self, X):
        """Add a bias term to the input features."""
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y, method='closed_form'):
        """Fit the linear regression model to the training data.

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target values.
            method (str): Method for fitting the model ('closed_form', 'lms',
                          'lwr')
        """
        X = self._add_bias(X)

        if method == 'closed_form':
            self._fit_closed_form(X, y)
        elif method == 'lms':
            self._fit_lms(X, y)
        elif method == 'lwr':
            self._fit_lwr(X, y)
        else:
            raise ValueError(
                "Invalid method. Use 'closed_form', 'lms', or 'lwr'.")

    def _fit_closed_form(self, X, y):
        """Fit the model using the closed-form solution."""
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def _fit_lms(self, X, y, learning_rate=0.01, epochs=100):
        """Fit the model using Least Mean Squares (LMS).

        Args:
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of iterations.
        """
        self.theta = np.zeros(X.shape[1])

        for _ in range(epochs):
            error = y - X @ self.theta
            gradient = -2 * X.T @ error
            self.theta -= learning_rate * gradient

    def _fit_lwr(self, X, y, tau=0.1):
        """Fit the model using Locally Weighted Regression (LWR).

        Args:
            tau (float): Bandwidth parameter.
        """
        self.theta = np.zeros(X.shape[1])

        for i in range(X.shape[0]):
            weights = np.exp(-(np.sum((X - X[i])**2, axis=1)) / (2 * tau**2))
            W = np.diag(weights)
            self.theta += np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

    def predict(self, X):
        """Make predictions using the trained model.

        Args:
            X (numpy.ndarray): Input features for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        X = self._add_bias(X)
        return X @ self.theta


def generate_synthetic_data():
    """Generate synthetic data for linear regression."""
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def main():

    # Generate synthetic data
    X, y = generate_synthetic_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]

    # Initialize and fit the Linear Regression model using the closed-form
    # solution
    model = LinearRegression()
    model.fit(X_train, y_train, method='closed_form')

    # Make predictions on the test set
    predictions = model.predict(X_test)
    # print("Matplotlib backend:", matplotlib.get_backend())

    # Set rcParams for font size
    plt.rcParams['font.size'] = 14  # Font size for labels (xlabel, ylabel)
    plt.rcParams['xtick.labelsize'] = 12  # Font size for x-axis tick labels
    plt.rcParams['ytick.labelsize'] = 12  # Font size for y-axis tick labels
    plt.rcParams['legend.fontsize'] = 14  # Font size for legend

    # Plot the original data points and the regression line
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    ax.scatter(
        X_test,
        y_test,
        color='#1f77b4',  #'#3883C9',
        label='Actual Data',
        alpha=0.75)
    ax.plot(
        X_test,
        predictions,
        color='#d62728',  #'#C62828',
        linewidth=3,
        label='Regression Line',
        alpha=0.75)
    ax.set_title('Linear Regression using Closed-Form Solution')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get the figure folder from the environment variable or use a default value
    figure_folder = os.environ.get('FIGURE_FOLDER', '/app/figures/')

    # Save the figure using the specified folder
    # plt.savefig(os.path.join(figure_folder, 'output.png'))
    plt.savefig(os.path.join('/app/figures/', 'linear_regression_fit.png'))

    try:
        plt.show()
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()
