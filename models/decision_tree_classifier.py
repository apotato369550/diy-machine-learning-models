import numpy as np

class SimpleDecisionTreeClassifier:
    def __init__(self, max_depth=5, min_sample_split=2):
        # max_depth = how deep the tree can grow before we stop splitting
        # min_samples_split = minimum number of samples needed to even try splitting a node
        # tree_ = where we'll store the actual decision tree structure once it's built
        self.max_depth = max_depth
        self.min_samples_split = min_sample_split
        self.tree_ = None

    def _gini(self, y):
        # y = the target labels for a group of samples
        # classes = unique values in y (like 0s and 1s for binary classification)
        # counts = how many times each class appears
        # prob = probability of each class (counts divided by total samples)
        # compute gini impurity
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def _best_split(self, X, y):
        """find the best feature + threshold to split on, using gini gain."""
        # X = feature matrix (rows = samples, columns = features)
        # y = target labels for each sample
        # best_feat = index of the feature that gives best split
        # best_thresh = value to split that feature on
        # best_gain = how much impurity we reduce with this split
        # current_impurity = gini score before any split (baseline)
        # n_features = number of columns in X
        # feature = current feature we're testing
        # thresholds = unique values in this feature (possible split points)
        # thresh = current threshold value we're trying
        # left_mask/right_mask = boolean arrays showing which samples go left or right
        # left_impurity/right_impurity = gini scores for each side of split
        # n = total number of samples
        # weighted_impurity = average impurity after split (weighted by sample count)
        # gain = how much cleaner the split makes things (current - weighted)
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
        # X = feature matrix for current group of samples
        # y = target labels for current samples
        # depth = how deep in the tree we currently are

        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth) or (num_labels == 1) or (num_samples < self.min_samples_split):
            leaf_value = np.bincount(y).argmax()
            return {"leaf": leaf_value}
        
        feat, thresh = self._best_split(X, y)

        if feat is None:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": leaf_value}
        
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feat,
            "threshold": thresh,
            "left": left_tree,
            "right": right_tree,
        }
    
    def fit(self, X, y):
        # X = training feature matrix
        # y = training target labels
        # tree_ = stores the built decision tree
        self.tree_ = self._build_tree(X, y)

    def _predict_one(self, x, node):
        # x = single sample's feature values
        # node = current node in the tree we're traversing
        if 'leaf' in node:
            return node['leaf']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        # X = feature matrix for samples we want to predict
        return np.array([self._predict_one(x, self.tree_) for x in X])
    
    def score(self, X, y):
        # X = feature matrix for test samples
        # y = true labels for test samples
        # preds = our model's predictions for these samples
        preds = self.predict(X)
        return np.mean(preds == y)