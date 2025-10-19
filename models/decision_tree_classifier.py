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
        """find the best feature + threshold to split on, using gini gain."""
        # initialize bests: none found yet, and best_gain starts negative so any real gain replaces it
        best_feat, best_thresh, best_gain = None, None, -1
        # impurity before splitting (baseline to compare against)
        current_impurity = self._gini(y)

        n_features = X.shape[1]
        # iterate over every feature to see where a split helps most
        for feature in range(n_features):
            # try unique values of this feature as candidate thresholds
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                # boolean masks for left/right split based on threshold
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask

                # skip degenerate splits that leave one side empty
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                # compute impurity for each side of the split
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                n = len(y)
                # weighted average impurity after the split
                weighted_impurity = (len(y[left_mask]) / n) * left_impurity + (len(y[right_mask]) / n) * right_impurity

                # information gain = decrease in impurity
                gain = current_impurity - weighted_impurity

                # if this split gives more gain than previous best, remember it
                if gain > best_gain:
                    best_feat, best_thresh, best_gain = feature, thresh, gain

        # return the index of the best feature and the threshold value (None if no valid split)
        return best_feat, best_thresh
    
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