import re
from models.logistic_regression import sigmoid

def sex_mapping(sex):
    sex_map = {'male': 0, 'female': 1}
    return sex_map[sex] if sex else None

def predict(X, w, b):
    z = X @ w + b
    probabilities = sigmoid(z)
    
    predictions = (probabilities >= 0.5).astype(int)
    
    return predictions

def extract_title(name):
    match = re.search(r',\s*(Mr|Mrs|Miss|Master|Ms|Don|Dr|Rev|Mme|Major|Lady|Sir|Mlle|Col|Capt|the Countess|Jonkheer)\.', name)
    return match.group(1) if match else None

def group_titles(title):
    if title == 'Mr':
        return 'Adult_Male'
    elif title in ['Mrs', 'Mme']:  
        return 'Married_Female'
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 'Single_Female'  
    elif title == 'Master':
        return 'Child_Male'
    elif title in ['Lady', 'Sir', 'the Countess', 'Don', 'Jonkheer']:
        return 'Nobility'
    elif title in ['Dr', 'Col', 'Major', 'Capt']:
        return 'Professional'
    elif title == 'Rev':
        return 'Clergy'
    else:
        return 'Other'

def group_mapping(group):
    group_map = {
        'Adult_Male': 0,
        'Married_Female': 1, 
        'Single_Female': 2,
        'Child_Male': 3,
        'Nobility': 4,
        'Professional': 5,
        'Clergy': 6,
        'Other': 7
    }

    return group_map[group]
