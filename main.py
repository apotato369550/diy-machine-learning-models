from utils import make_linear

X, y = make_linear(100, 7, 4.0, 6, 42069)

# first 5 features/samples
print(X[:5])

# first 5 labels
print(y[:5])