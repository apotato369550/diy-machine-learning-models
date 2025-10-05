import numpy as np

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return
    
    def fit(self, X, y):
        return
    
    def predict_proba(self, X):
        return
    
    def predict(self, X, threshold=0.5):
        return
    
    def score(self, X, y):
        return