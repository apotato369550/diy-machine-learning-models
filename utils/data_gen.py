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

def make_logistic(n_samples=100, n_features=1, coef=None, intercept=0.0, noise=0.1, random_state=None):
    # generate random normal dist
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))

    # set coefficients if none specified
    if coef is None:
        coef = rng.uniform(-2, 2, size=n_features)

    # linear combination. dot product of x and coef matrix
    linear_combination = X.dot(coef) + intercept

    # get sigmoid for probabilities
    probs = 1 / (1 + np.exp(-linear_combination))

    # get y sample from bernoulli distribution
    y = rng.binomial(1, probs)

    if noise > 0:
        flip_mask = rng.random(n_samples) < noise
        y[flip_mask] = 1 - y[flip_mask]

    return X, y, coef, intercept

def make_decision_tree(n_samples=200, n_features=2, random_state=None, noise=0.1):
    '''
    create a dataset for decision tree classification
    '''

    rng = np.random.default_rng(random_state)

    X = rng.uniform(-1, 1, size=(n_samples, n_features))

    y = np.zeros(n_samples, dtype=int)
    rule = (X[:, 0] + X[:, 1] > 0) & (X[:, 0] > -0.3)
    y[rule] = 1

    if noise > 0:
        flip_mask = rng.random(n_samples) < noise
        y[flip_mask] = 1 - y[flip_mask]

    return X, y

def make_clusters(n_samples=200, n_features=2, n_clusters=3, cluster_std=0.3, random_state = None):
    rng = np.random.default_rng(random_state)
    centers = rng.uniform(-5, 5, size=(n_clusters, n_features))
    X = []
    y = []

    for i, center in enumerate(centers):
        points = center + rng.normal(scale=cluster_std, size=(n_samples // n_clusters, n_features))
        X.append(points)
        y.append(np.full(points.shape[0], i))

    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y, centers


def make_naive_bayes_data(n_samples=200, n_features=3, random_state=None):
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n_samples) > 0).astype(int)

    return X, y

def make_knn_data(n_samples=200, n_features=2, n_classes=2, cluster_std=0.8, random_state=None):
    rng = np.random.default_rng(random_state)

    centers = rng.uniform(-5, -5, size=(n_classes, n_features))

    samples_per_class = [n_samples // n_classes] * n_classes

    for i in range(n_samples % n_classes):
        samples_per_class[i] += 1

    X = []
    y = []

    for class_idx, n_class_samples in enumerate(samples_per_class):
        cluster = rng.normal(
            loc=centers[class_idx],
            scale=cluster_std,
            size=(n_class_samples, n_features)
        )
        X.append(cluster)
        y.append(np.full(n_class_samples, class_idx))

    X = np.vstack(X)
    y = np.concatenate(y)

    indices = rng.permutation(n_samples)
    return X[indices], y[indices], centers