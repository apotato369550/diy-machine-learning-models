from utils import make_linear, train_test_split
from utils import mse, rmse, mae, r2_score
from models import SimpleLinearRegression
import matplotlib.pyplot as plt

X, y = make_linear(100, 3, 4.0, 6, 42069)

# first 5 features/samples
print(X[:5])

# first 5 labels
print(y[:5])

# let's do "eda" next lol

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X_train[:5])
print(X_train.shape)

print(X_test[:5])
print(X_test.shape)

print(y_train[:5])
print(y_train.shape)

print(y_test[:5])
print(y_test.shape)

# figure out how to add line of best fit (linear regression line) into visualization

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Train")
plt.scatter(X_test, y_test, color="green", alpha=0.6, label="Test")

plt.title("Simple Linear Regression Fit")
plt.xlabel("X feature")
plt.ylabel("y target")
plt.legend()
plt.show()


linreg = SimpleLinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
r_squared = linreg.score(X_test, y_test)

# figure out how to interpret this, then maybe confusion matrix probably
# metrics.py
print("r-squared score: " + str(r_squared))

# scoringgg
print("MSE: " + str(mse(y_test, y_pred)))
print("RMSE: " + str(rmse(y_test, y_pred)))
print("MAE: " + str(mae(y_test, y_pred)))
print("R2 Score: " + str(r2_score(y_test, y_pred)))