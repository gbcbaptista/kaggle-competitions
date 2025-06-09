import numpy as np

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
    w = w_init.copy()
    b = b_init

    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            J_history.append(cost)
            print(f"Iteration {i:4d}: Cost {cost:.6f}")
    
    return w, b, J_history