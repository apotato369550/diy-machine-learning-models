import numpy as np

class SimpleGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.means = np.zeros((len(self.classes), n_features))
        self.vars = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for index, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[index, :] = X_c.mean(axis=0)
            self.vars[index, :] = X_c.var(axis=0) + 1e-9
            self.priors[index] = X_c.shape[0] / X.shape[0]
    
    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _posterior(self, x):
        return
    
    def predict(self, X):
        return