from utils.data_gen import make_clusters
from models.simple_kmeans_classifier import KMeans
import matplotlib.pyplot as plt

X, y_true, centers = make_clusters(n_samples=300, n_clusters=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_