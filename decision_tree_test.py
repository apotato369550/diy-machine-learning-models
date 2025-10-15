from utils.data_gen import make_decision_tree
import matplotlib.pyplot as plt

X, y = make_decision_tree(n_samples=200, random_state=42)

plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Decision Tree Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()