import numpy as np

def make_linear(n_samples=50, noise=5, coef=5.0, intercept=45.0, seed=None):
    """
    Generate a simple linear regression dataset.

    Params:

    n_samples : int - number of data points to generate
    noise : float - standard deviation of noise added to y
    coef : float - true slope (beta1)
    intercept : float - true intercept (beta0)
    seed : int or None - random seed for reproducibility

    Returns
    X, y
    X : np.ndarray - returns feature values of the shape (n_samples, 1)
    y : np.ndarray - returns target values of the shape (n_samples, )

    """

    # instantiate a random no generator
    random_generator = np.random.default_rng(seed)

    # generate random feature values between 0 and 10
    # n_samples rows, 1 column
    X = random_generator.uniform(0, 10, size=(n_samples, 1))

    y_true = intercept + coef * X.flatten()
    noise_values = random_generator.normal(0, noise, size=n_samples)
    y = y_true + noise_values

    return X, y