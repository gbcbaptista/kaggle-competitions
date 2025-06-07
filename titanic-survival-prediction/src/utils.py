from models.logistic_regression import sigmoid

def predict(X, w, b):
    z = X @ w + b
    probabilities = sigmoid(z)
    
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions

