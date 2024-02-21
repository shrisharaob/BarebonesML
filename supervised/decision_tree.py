"""Decision Tree

Algorithm
---------
 - Selecting the Best Splits: Identify the feature and value that best separates
   the data
 - Building Nodes: Create nodes based on the splits
 - Recursive Splitting: Repeat steps bove recursively for each subset until
   stopping criteria are met
 - Stopping Criteria: Define conditions to stop splitting, such as reaching a
   maximum depth or minimum sample size
 - Prediction: Traverse the tree to predict the label of new instances based on
   learned rules
"""
import numpy as np
from collections import Counter
from pathlib import Path
import sys
# cwd
directory = Path(__file__).absolute()
# setting path
sys.path.append(str(directory.parent.parent))
from utils.train_test_split import train_test_split
from utils.generate_toy_data import generate_data, generate_scaled_data
from utils.visualizations import plot_data_with_labels


class Node(object):
    """Nodes
    """

    def __init__(self,
                 left_node=None,
                 right_node=None,
                 value=None,
                 feature_idx=None,
                 threshold=None):
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.feature_idx = feature_idx
        self.threshold = threshold

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree(object):
    """Documentation for DecisionTree

    """

    def __init__(self, max_depth=100, min_samples=2, min_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_features = min_features
        self.tree = None

    def fit(self, X, y):
        if self.min_features is None:
            self.min_features = X.shape[1]
        else:
            self.min_features = min(X.shape[1], self.min_features)
        self.tree = self.grow_tree(X, y)

    def predict(self, X):
        # row wise
        print(X.shape)
        return np.array([self.traverse_tree(row, self.tree) for row in X])

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # check stopping conditions
        condition_1 = depth > self.max_depth
        condition_2 = n_samples < self.min_samples
        condition_3 = n_labels == 1
        if condition_1 or condition_2 or condition_3:
            # print('stopping reached')
            # value = self.get_node_value(y)
            # print(value)
            return Node(value=self.get_node_value(y))

        # randomly pick #min_features cols
        feature_idxs = np.random.choice(n_features,
                                        self.min_features,
                                        replace=False)

        # search best splits for the randomly picked feature cols
        best_feature_idx, best_threshold = self.search_best_split(
            X, y, feature_idxs)

        # grow leaves
        # left_idxs = X[:, best_feature_idx] <= best_threshold
        # right_idxs = np.logical_not(left_idxs)
        left_idxs, right_idxs = self.split_function(X[:, best_feature_idx],
                                                    best_threshold)

        left_node = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_node = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(left_node=left_node,
                    right_node=right_node,
                    feature_idx=best_feature_idx,
                    threshold=best_feature_idx)

    def split_function(self, X_selected, threshold):
        """Returns the booleen indices of left and right split of elements in
        the feature column based on the threshold provided 
        """
        left_idxs = X_selected <= threshold
        right_idxs = np.logical_not(left_idxs)
        return left_idxs, right_idxs

    def compute_entropy(self, x):
        """Returns entropy defined as:
        entropy = -1 *  Σ P(x) log_2(P(x))
        """
        # cnts = np.bincount(x)
        values, cnts = np.unique(x, return_counts=True)
        prob = cnts / len(x)  # P(x)
        return -1.0 * sum([p_i * np.log2(p_i) for p_i in prob if p_i > 0])

    def compute_information_gain(self, X, y, threshold):
        """Returns information gain defined as:
        IG = H(parent) - w_i * H(child_i)
        i.e. entropy of parent minus the weighted avg entropy of all its children
        """
        entropy_of_parent = self.compute_entropy(y)
        left_idxs, right_idxs = self.split_function(X, threshold)
        num_samples_in_left = sum(left_idxs)
        num_samples_in_right = sum(right_idxs)
        num_samples_in_parent = len(y)
        weight_left = num_samples_in_left / num_samples_in_parent
        weight_right = num_samples_in_right / num_samples_in_parent
        avg_entropy_of_children = weight_left * \
            self.compute_entropy(y[left_idxs]) +\
            weight_right * self.compute_entropy(y[right_idxs])
        information_gain = entropy_of_parent - avg_entropy_of_children
        return information_gain

    def get_node_value(self, y):
        cntr = Counter(y)
        return cntr.most_common(1)[0][0]

    def search_best_split(self, X, y, feature_idxs):
        """search all possible splits for the best split based on information
        gain 
        """
        max_gain, best_feature_idx, best_threshold = -1, None, None
        # get the integer values of the features col idx
        feature_int_idxs = np.argwhere(feature_idxs).flatten()
        for f_idx in feature_int_idxs:
            X_selected = X[:, f_idx]
            possible_split_thresholds = np.unique(X_selected)
            for threshold in possible_split_thresholds:
                info_gain = self.compute_information_gain(
                    X_selected, y, threshold)
                if info_gain > max_gain:
                    best_feature_idx = f_idx
                    best_threshold = threshold
                    max_gain = info_gain
        #
        return best_feature_idx, best_threshold

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.traverse_tree(x, node.left_node)
        return self.traverse_tree(x, node.right_node)


def visualize_tree(node, depth=0, indent="   "):
    print('-' * 10)
    if node is None:
        return

    if node.is_leaf_node():
        print(indent * depth + "Leaf Node: Predicted Value =", node.value)
    else:
        print(indent * depth + "Feature Index:", node.feature_idx,
              "Threshold:", node.threshold)
        print(indent * depth + "Left Branch:")
        visualize_tree(node.left_node, depth + 1, indent)
        print(indent * depth + "Right Branch:")
        visualize_tree(node.right_node, depth + 1, indent)


def main():
    X, y = generate_scaled_data()
    # y[y < 0] = 2
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    # fit
    tree = DecisionTree(max_depth=100)
    tree.fit(X_train, y_train)

    # visualize_tree(tree.tree)

    preds = tree.predict(X_train)
    plot_data_with_labels(X_train, y_train, preds)

    # preds = tree.predict(X_test)
    # plot_data_with_labels(X_test, y_test, preds)


if __name__ == '__main__':
    main()
