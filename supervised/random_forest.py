# Random Forest
import numpy as np
from decision_tree import DecisionTree
from pathlib import Path
import sys

directory = Path(__file__).absolute()  # cwd
sys.path.append(str(directory.parent.parent))  # setting path
from utils.train_test_split import train_test_split  # noqa
from utils.generate_toy_data import generate_data, generate_scaled_data  # noqa
from utils.visualizations import plot_data_with_labels  # noqa


class RandomForest(object):
    """Documentation for RandomForest

    """

    def __init__(self, num_trees=10, max_depth=100):
        # super(RandomForest, self).__init__()
        self.num_trees = num_trees
        self.max_depth = max_depth
