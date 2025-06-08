import re
from pathlib import Path
import pandas as pd
from models.logistic_regression import sigmoid

def load_data():
    current_dir = Path(__file__).parent
    
    train_path = current_dir / '../data/raw/train.csv'
    test_path = current_dir / '../data/raw/test.csv'
    
    train_path = train_path.resolve()
    test_path = test_path.resolve()
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


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

def impute_null_age_strategict(df):
    df_copy = df.copy()
    df_copy['Title'] = df_copy['Name'].map(extract_title)
    median_age_per_title = df_copy.groupby('Title')['Age'].median()

    df_copy['Age'] = df_copy.apply(lambda row: median_age_per_title[row['Title']] if pd.isnull(row['Age']) else row['Age'], axis=1)
    df_copy['Age'].isnull().sum()

    return df_copy
