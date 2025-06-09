import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_  = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = X @ w + b
    f_wb = sigmoid(z)

    cost = -(y @ np.log(f_wb) + (1-y) @ np.log(1-f_wb))/m
    return cost

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    z = X @ w + b
    f_wb = sigmoid(z)
    error = f_wb - y

    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m

    return dj_dw, dj_db

def gradient_descent(X, y, w_init, b_init, alpha, num_iters):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    w = w_init.copy()
    b = b_init

    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X_scaled, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0:
            cost = compute_cost(X_scaled, y, w, b)
            J_history.append(cost)
            print(f"Iteration {i:4d}: Cost {cost:.6f}")
    
    return w, b, J_history