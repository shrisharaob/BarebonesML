"""This module assumes a Gaussian distribution for the features. The fit method
estimates the mean and standard deviation for each feature in each class, and
the predict method calculates the likelihoods based on Gaussian distribution and
predicts the class with the maximum likelihood.
"""
import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier

    Parameters:
    -----------
    None

    Attributes:
    -----------
    class_probs : array, shape (n_classes,)
        Prior probabilities of each class.

    feature_probs : array, shape (n_classes, n_features, 2)
        Probability distributions of features for each class.
        The third dimension holds mean and standard deviation.

    Methods:
    -----------
    fit(X, y)
        Fit the Naive Bayes classifier to the training data.

    predict(X)
        Predict the class labels for input data.

    Examples:
    -----------
  
    """

    def __init__(self):
        self.class_probs = None
        self.feature_probs = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.

        Returns:
        -----------
        None
        """
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        class_probs = np.zeros(n_classes)
        feature_probs = np.zeros((n_classes, n_features, 2))

        for i, c in enumerate(unique_classes):
            class_mask = (y == c)
            class_probs[i] = np.sum(class_mask) / n_samples

            for j in range(n_features):
                feature_probs[i, j, 0] = np.mean(X[class_mask, j])
                feature_probs[i, j, 1] = np.std(X[class_mask, j])

        self.class_probs = class_probs
        self.feature_probs = feature_probs

    def predict(self, X):
        """
        Predict the class labels for input data.

        Parameters:
        -----------
        X : array, shape (n_samples, n_features)
            Input data.

        Returns:
        -----------
        predictions : array, shape (n_samples,)
            Predicted class labels.
        """
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            likelihoods = np.zeros(len(self.class_probs))

            for j, class_prob in enumerate(self.class_probs):
                class_likelihood = np.prod(
                    (1 / (np.sqrt(2 * np.pi) * self.feature_probs[j, :, 1])) *
                    np.exp(-(X[i] - self.feature_probs[j, :, 0])**2 /
                           (2 * self.feature_probs[j, :, 1]**2)))
                likelihoods[j] = class_prob * class_likelihood

            predictions[i] = np.argmax(likelihoods)

        return predictions


X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

X_test = np.array([[2, 3], [4, 5]])
predictions = nb_classifier.predict(X_test)

print(predictions)
