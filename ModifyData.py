import random
import pandas as pd
import numpy as np
import challenge_basic

# file_name = "clean_dataset.csv"
file_name = "pre_data.csv"
# From lab06
def make_onehot(indicies):
    I = np.eye(4)
    return I[indicies]

# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(z):
    return np.exp((z - np.max(z))) / np.sum(np.exp(z), axis=0)

def splitDataset(df, val_size, test_size):
    train_size = len(df) - (val_size + test_size)
    random.shuffle(df)
    D_train = df[:train_size]
    D_valid = df[train_size:train_size + val_size]
    D_test = df[-test_size:]

    t_train = D_train['Label']
    X_train = D_train.drop('Label', axis=1)
    t_valid = D_valid['Label']
    X_valid = D_valid.drop('Label', axis=1)
    t_test = D_test['Label']
    X_test = D_test.drop('Label', axis=1)
    
    return X_train, t_train, X_valid, t_valid, X_test, t_test

if __name__ == "__main__":
    df = pd.read_csv(file_name).dropna()
    df["Q1"] = df["Q1"].apply(challenge_basic.get_number)
    df["Q2"] = df["Q2"].apply(challenge_basic.get_number)
    df["Q3"] = df["Q3"].apply(challenge_basic.get_number)
    df["Q4"] = df["Q4"].apply(challenge_basic.get_number)
    # Add codes
    df["Q7"] = df["Q7"].apply(challenge_basic.to_numeric)
    df["Q8"] = df["Q8"].apply(challenge_basic.to_numeric)
    df["Q9"] = df["Q9"].apply(challenge_basic.to_numeric)
    Q1_onehot = pd.get_dummies(df['Q1'], prefix='Q1', dtype=int)
    Q2_onehot = pd.get_dummies(df['Q2'], prefix='Q2', dtype=int)
    Q3_onehot = pd.get_dummies(df['Q3'], prefix='Q3', dtype=int)
    Q4_onehot = pd.get_dummies(df['Q4'], prefix='Q4', dtype=int)
    
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"{cat}"
      df[cat_name] = df["Q5"].apply(lambda s: challenge_basic.cat_in_s(s, cat))

    Q6_categories = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']

    df['Q6_max'] = pd.Categorical(df['Q6_max'], categories=Q6_categories)
    Q6_max_onehot = pd.get_dummies(df['Q6_max'], prefix='Q6_max', dtype=int)
    df['Q6_min'] = pd.Categorical(df['Q6_min'], categories=Q6_categories)
    Q6_min_onehot = pd.get_dummies(df['Q6_min'], prefix='Q6_min', dtype=int)
    
    df = pd.concat([df, Q1_onehot, Q2_onehot, Q3_onehot, Q4_onehot, Q6_max_onehot, Q6_min_onehot], axis=1)

    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5_list', 'id', 'Q5', 'Q6', 'p', 'f', 's', 'o', 'Q10', 'Q6_max', 'Q6_min'] # Edit Accordingly
    for col in delete_columns:
        del df[col]
    
    df.to_csv("ModifiedData.csv", index=False)