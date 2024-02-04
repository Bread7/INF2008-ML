import numpy as np

from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a node class with 2 functions
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *,value=None):
        self.feature = feature # Specify feature
        self.threshold = threshold # Specify threshold of feature
        self.left =  left # Keep left subtree
        self.right =  right # Keep right subtree
        self.value = value # Keep value of node, None if it is a leaf

    # Boolean function. Return true if "value" parameter is not None
    def is_leaf_node(self):
        if (self.value != None):
            return True
        return False
    
# Decision Tree Class
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split =  min_samples_split # Minimum number of samples to contain before splitting
        self.max_depth = max_depth # Maximum depth of tree
        self.n_features = n_features # Number of features the tree will receive
        self.root = None

    # X is number of observations with shape at index 0, number of features with shape at index 1
    # Calls build_tree function with X and y to get root of tree
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.build_tree(X, y)
    
    def build_tree(self, X, y, depth=0):
        # Book keeping initialisation
        n, f = X.shape
        n_labels = len(np.unique(y)) 

        # Check stopping criteria. Return to root if stopping criteria reached
        if ((depth >= self.max_depth) or (n_labels == 1) or (n < self.min_samples_split)):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Find best split based on information gain
        # feat_idx = f
        feat_idx = np.random.choice(f, self.n_features, replace=False)

        # Split tree into left and right recursively
        best_feature, best_thresh= self.best_split(X, y, feat_idx)

        # Create child nodes
        left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)
        left = self.build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)


    def best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # if X_column not in thresholds else None
            for thr in thresholds:
                gain = self.information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def information_gain(self, y, X_column, threshold):
        left_idx, right_idx = self.split(X_column, threshold)

        if ((len(left_idx) < 1) or (len(right_idx) < 1)):
            return 0
        
        n = len(y)

        # Find number of observations on left / right subtree
        n_l, n_r = len(left_idx), len(right_idx)

        # Use entropy function to calculate amount of entropy [er branch
        e_l, e_r = self.entropy(y[left_idx]), self.entropy(y[right_idx])

        # Find amount of entropy in child
        # child_entropy = n + n_l + n_r + e_l + e_r
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = 1 - child_entropy
        return information_gain

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy

    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
data = datasets.load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=55)
    
clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
