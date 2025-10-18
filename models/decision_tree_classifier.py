import numpy as np

class SimpleDecisionTreeClassifier:
    def __init__(self, max_depth=5, min_sample_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_sample_split
        self.tree_ = None

    def _gini(self, y):
        # compute gini impurity
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def _best_split(self, X, y):
        # find the best feature and threshold to split on
        return
    
    def _build_tree(self, X, y, depth=0):
        return 
    
    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if 'leaf' in node:
            return node['leaf']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])
    
    def score(self, X, y):
        preds = self.Predict(X)
        return np.mean(preds == y)