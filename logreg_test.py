from utils.data_gen import make_logistic
from utils import train_test_split
from models.simple_logistic_regression import SimpleLogisticRegression
import matplotlib.pyplot as plt

# create toy dataset
X, y, coef, intercept = make_logistic(
    n_samples=200,
    n_features=2,
    random_state=42
)

# print stats of data
print("X shape: ", X.shape)
print("y distribution: ", {0: sum(y==0), 1:sum(y==1)})
print("True coef: ", coef)
print("True intercept: ", intercept)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Train")
plt.scatter(X_test, y_test, color="green", alpha=0.6, label="Test")

plt.title("Simple Linear Regression Fit")
plt.xlabel("X feature")
plt.ylabel("y target")
plt.legend()
plt.show()

# initialize model and train
logreg = SimpleLogisticRegression(learning_rate=0.1, n_iters=2000, verbose=True)
logreg.fit(X_train, y_train)

# evaluate accuracy
accuracy = logreg.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.3f}")
print("Learned coefficients: ", logreg.coef_)
print("Learned intercept: ", logreg.intercept_)
