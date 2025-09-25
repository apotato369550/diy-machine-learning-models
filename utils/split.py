import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    X = np.array(X)
    y = np.array(y)

    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    test_size = int(n_samples * test_size)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    return X_train, X_test, y_train, y_test