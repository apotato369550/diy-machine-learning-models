from utils.data_gen import make_logistic
from utils import train_test_split
from models.simple_logistic_regression import SimpleLogisticRegression

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

# initialize model and train
logreg = SimpleLogisticRegression(learning_rate=0.1, n_iters=1000, verbose=True)
logreg.fit(X_train, y_train)

# evaluate accuracy
accuracy = logreg.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.3f}")
print("Learned coefficients: ", logreg.coef_)
print("Learned intercept: ", logreg.intercept_)
