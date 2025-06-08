import re
from models.logistic_regression import sigmoid

def predict(X, w, b):
    z = X @ w + b
    probabilities = sigmoid(z)
    
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions

def extract_title(name):
    match = re.search(r',\s*(Mr|Mrs|Miss|Master|Ms|Don|Dr|Rev|Mme|Major|Lady|Sir|Mlle|Col|Capt|the Countess|Jonkheer)\.', name)
    return match.group(1) if match else None