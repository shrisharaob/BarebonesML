# Random Forest
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


class RandomForest(object):
    """RandomForest(num_trees=10,min_samples=2,max_depth=100,min_features=None)
    """

    def __init__(self,
                 num_trees=10,
                 min_samples=2,
                 max_depth=100,
                 min_features=None):
        # super(RandomForest, self).__init__()
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_features = min_features
        self.trees = []  # forest made of a list of trees!!!!

    def fit(self, X, y):
        """fit(X, y):
        """
        for i in range(self.num_trees):
            ith_tree = DecisionTree(max_depth=self.max_depth,
                                    min_samples=self.min_samples,
                                    min_features=self.min_features)
            X_resampled, y_resampled = self.bootstrap(X, y)
            ith_tree.fit(X_resampled, y_resampled)
            self.trees.append(ith_tree)

    def bootstrap(self, X, y, resample_fraction=1):
        num_samples = X.shape[0]
        if resample_fraction > 1:
            resample_fraction = 1.0
        resample_size = int(num_samples * resample_fraction)
        idxs = np.random.choice(num_samples, resample_size, replace=True)
        return X[idxs], y[idxs]

    def agg_func(self, y):
        cntr = Counter(y)
        return cntr.most_common(1)[0][0]

    def predict(self, X):
        preds = [tree.predict(X)
                 for tree in self.trees]  # dims: n_trees x n_samples
        preds = np.swapaxes(preds, 0, 1)  # dims: n_samples x n_trees
        final_preds = np.array([self.agg_func(p) for p in preds])
        return final_preds


def main():
    X, y = generate_data(num_samples=100)
    # y[y < 0] = 2
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    y_train[y_train == -1] = 0
    print(np.unique(y_train))

    # fit
    rf = RandomForest(max_depth=100, num_trees=100)
    rf.fit(X_train, y_train)

    # visualize_tree(tree.tree)

    preds = rf.predict(X_train)
    plot_data_with_labels(X_train, y_train, preds)

    # preds = tree.predict(X_test)
    # plot_data_with_labels(X_test, y_test, preds)


if __name__ == '__main__':
    main()
