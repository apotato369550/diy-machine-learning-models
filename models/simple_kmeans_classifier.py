import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, n_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)

        # randomly pick centroids
        random_idx = rng.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.n_iters):
            #assign clusters
            labels = self._assign_clusters(X)

            # update the centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
    
            # stop if converged
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
    
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        return self._assign_clusters(X)
