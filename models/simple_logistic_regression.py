import numpy as np

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, fit_intercept=True, verbose=False):
        # step size for gradient descent
        self.learning_rate = learning_rate

        # number of iterations for training
        self.n_iters = n_iters

        # whether to include an intercept term
        self.fit_intercept = fit_intercept

        # if true, prints loss every 100 iterations
        self.verbose = verbose

        # idk what this is yet
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        # sigmoid activation
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # fit the logistic regression model using gradient descent
        # (really need to look into what gradient descent is lol)
        X = np.array(X)
        y = np.array(y)

        # reshape y if necessary
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        n_samples, n_features = X.shape

        # initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0 if self.fit_intercept else None

        # training loop i presume
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.coef_) + (self.intercept_ if self.fit_intercept else 0)

            # predicted probabilities
            y_pred = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y) if self.fit_intercept else 0

            # update parameters based on gradients
            self.coef_ -= self.learning_rate * dw
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate *db

            # print loss if necessary
            if self.verbose and i % 100 == 0:
                loss -= -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
                print(f"Iteration {i}: Loss = {loss:.4f}")
        
        return self

    
    def predict_proba(self, X):
        # predict class probabilities
        X = np.array(X)
        linear_model = np.dot(X, self.coef_) + (self.intercept_ if self.fit_intercept else 0)
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        # predict binary class labels 0 or 1
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)