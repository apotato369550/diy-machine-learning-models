from utils.data_gen import make_naive_bayes_data
from models.simple_gaussian_naive_bayes import SimpleGaussianNB
from utils import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = make_naive_bayes_data(n_samples=200, n_features=2, random_state=42)

print("X shape: ", X.shape)
print("y distribution: ", {0: np.sum(y == 0), 1: np.sum(y == 1)})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# visualize dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", label="Train", alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", marker="x", label="Test", alpha=0.7)
plt.title("Naive Bayes Classification Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()