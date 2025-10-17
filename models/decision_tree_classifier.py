import numpy as np

class SimpleDecisionTreeClassifier:
    def __init__(self, max_depth=5, min_sample_split=2):
        return
    
    def _gini(self, y):
        # compute gini impurity
        return

    def _best_split(self, X, y):
        # find the best feature and threshold to split on
        return
    
    def _build_tree(self, X, y, depth=0):
        return 
    
    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y)

    def _predict_one(self, x, node):
        return
    
    def predict(self, X):
        return
    
    def score(self, X, y):
        return