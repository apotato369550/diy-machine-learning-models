from utils.data_gen import make_clusters
from models.simple_kmeans_classifier import KMeans
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

X, y_true, centers = make_clusters(n_samples=300, n_clusters=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='*', s=300, c='black', alpha=1)
plt.title("K-Means Clustering Results")
plt.legend()
plt.savefig("plots/kmeans_clustering.png")
plt.show()
plt.close()