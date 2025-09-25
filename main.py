from utils import make_linear, train_test_split

X, y = make_linear(100, 7, 4.0, 6, 42069)

# first 5 features/samples
print(X[:5])

# first 5 labels
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X_train[:5])
print(X_train.shape)

print(X_test[:5])
print(X_test.shape)

print(y_train[:5])
print(y_train.shape)

print(y_test[:5])
print(y_test.shape)