from utils.data_gen import make_naive_bayes_data
from models.simple_gaussian_naive_bayes import SimpleGaussianNB
from utils import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

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
plt.savefig("plots/naive_bayes_data.png")
plt.show()
plt.close()

model = SimpleGaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy}")


# optional: visualize decision boundary (for 2D)
if X.shape[1] == 2:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolor="k")
    plt.title("Naive Bayes Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("plots/naive_bayes_decision_boundary.png")
    plt.show()
    plt.close()