from utils.data_gen import make_logistic

X, y, coef, intercept = make_logistic(
    n_samples=200,
    n_features=2,
    random_state=42
)

print("X shape: ", X.shape)
print("y distribution: ", {0: sum(y==0), 1:sum(y==1)})
print("True coef: ", coef)
print("True intercept: ", intercept)

