import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        '''
        A simple linear regression model that uses OLS (ordinary least squares)
        accepts:
        fit_intercept : bool - whether to calculate the bias (intercept) for the model. defaults to true
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        '''
        fit the linear model given features (training data) and a target (output values)

        X : training data in the shape of (n_samples, n_features)
        y : target values. 1d column (n_samples,)
        '''

        X = np.array(X)
        y = np.array(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # ols solution (gonna have to study this further)
        XTX = X.T.dot(X)
        XTy = X.T.dot(y)

        # our beta
        beta = np.linalg.inv(XTX).dot(XTy)

        # extract the intercept + coefficients
        if self.fit_intercept:
            self.intercept_ = beta[0, 0]
            self.coef_ = beta[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.flatten()
    
    def predict(self, X):
        '''
        method that actually predicts given the model

        X : samples. should be in the shape (n_samples, n_features)
        '''
        X = np.array(X)

        if self.fit_intercept:
            return X.dot(self.coef_) + self.intercept_
        else:
            return X.dot(self.coef_)
    
    def score(self, X, y):
        '''
        returns the score in r-squared (R^2)

        X and y
        '''
        y = np.array(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot