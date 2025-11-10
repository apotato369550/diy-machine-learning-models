from utils.data_gen import make_decision_tree
from utils import train_test_split
from models.decision_tree_classifier import SimpleDecisionTreeClassifier
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

X, y = make_decision_tree(n_samples=200, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

decision_tree = SimpleDecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, y_train)

accuracy = decision_tree.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# decision tree dataset
plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Decision Tree Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("plots/decision_tree_dataset.png")
plt.close()
