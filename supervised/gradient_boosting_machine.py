import numpy as np
from decision_tree import DecisionTree
from collections import Counter
from pathlib import Path
import sys

directory = Path(__file__).absolute()  # cwd
sys.path.append(str(directory.parent.parent))  # setting path
from utils.train_test_split import train_test_split  # noqa
from utils.generate_toy_data import generate_data, generate_scaled_data  # noqa
from utils.visualizations import plot_data_with_labels  # noqa


class GradientBoostingMachine:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initialize Gradient Boosting Machine with hyperparameters.
        
        Parameters:
        - n_estimators: Number of trees (estimators) in the ensemble.
        - learning_rate: The shrinkage parameter to prevent overfitting.
        - max_depth: Maximum depth of each decision tree in the ensemble.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []  # List to store the decision trees
        self.residuals = []  # List to store the residuals

    def fit(self, X, y):
        """
        Fit the Gradient Boosting Machine to the training data.
        
        Parameters:
        - X: Input features (numpy array or pandas DataFrame).
        - y: Target values (numpy array or pandas Series).
        """
        self.trees = []  # Reset the list of trees
        self.residuals = np.copy(
            y)  # Initialize residuals to the target values

        # Sequentially fit decision trees to the residuals
        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth)  # Create a decision tree regressor
            tree.fit(X, self.residuals)  # Fit the tree to the residuals
            self.trees.append(tree)  # Add the trained tree to the ensemble
            residuals_pred = tree.predict(X)  # Predictions of the current tree
            self.residuals = self.residuals.astype(float)
            self.residuals -= self.learning_rate * residuals_pred  # Update residuals using the learning rate

    def predict(self, X):
        """
        Make predictions using the fitted Gradient Boosting Machine.
        
        Parameters:
        - X: Input features for prediction (numpy array or pandas DataFrame).
        
        Returns:
        - Predicted target values (numpy array).
        """
        predictions = np.zeros(len(X))  # Initialize predictions to zeros

        # Combine predictions from all trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(
                X)  # Weighted sum of tree predictions
        return predictions


def main():
    X, y = generate_data(num_samples=100, seed=256)
    y[y == -1] = 0.0
    # y[y < 0] = 2
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # fit
    gbm = GradientBoostingMachine(max_depth=100, n_estimators=100)
    gbm.fit(X_train, y_train)

    # visualize_tree(tree.tree)

    preds = gbm.predict(X_train)
    plot_data_with_labels(X_train, y_train, preds)

    # preds = tree.predict(X_test)
    # plot_data_with_labels(X_test, y_test, preds)


if __name__ == '__main__':
    main()
