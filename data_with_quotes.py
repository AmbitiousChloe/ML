import random
import pandas as pd
import numpy as np
import CSC311_ML_Challenge.NeuralNetwork.challenge_basic as challenge_basic

file_name = "ModifiedData.csv"

import pandas as pd
import numpy as np
import random

def split_dataset(df: pd.DataFrame, val_size: int, test_size: int):
    df_shuffled = df.sample(frac=1, random_state=42)
    X = df_shuffled.drop(columns='Label').to_numpy()
    t = df_shuffled['Label'].to_numpy()
    
    total_size = len(df_shuffled)
    train_size = total_size - (val_size + test_size)
    
    X_train, t_train = X[:train_size], t[:train_size]
    X_valid, t_valid = X[train_size:train_size + val_size], t[train_size:train_size + val_size]
    X_test, t_test = X[-test_size:], t[-test_size:]
    
    return X_train, t_train, X_valid, t_valid, X_test, t_test


def radm_dict(d):
    value_to_keys = {}
    for key, value in d.items():
        if value in value_to_keys:
            value_to_keys[value].append(key)
        else:
            value_to_keys[value] = [key]

    new_dict = {}
    for value, keys in value_to_keys.items():
        if len(keys) > 1:
            key_to_keep = random.choice(keys)
            new_dict[key_to_keep] = value
        else:
            new_dict[keys[0]] = value
    
    return new_dict


def to_dict(s):
  samples = s.split(",")
  result_dict = {}
  for sample in samples:
    # Split the pair on "=>" to separate the key and value
    key, value = sample.split('=>')
    if value == "":
      break
    # Convert value to integer and add to the dictionary
    result_dict[key.strip()] = int(value.strip())
    result_dict = radm_dict(result_dict)
  return [max(result_dict, key=result_dict.get), min(result_dict, key=result_dict.get)]


if __name__ == "__main__":
    df = pd.read_csv(file_name).dropna()
    df["Q1"] = df["Q1"].apply(challenge_basic.get_number)
    df["Q2"] = df["Q2"].apply(challenge_basic.get_number)
    df["Q3"] = df["Q3"].apply(challenge_basic.get_number)
    df["Q4"] = df["Q4"].apply(challenge_basic.get_number)
    # Add codes
    df['Q6_max'] = df['Q6'].apply(lambda x: to_dict(x)[0])
    df['Q6_min'] = df['Q6'].apply(lambda x: to_dict(x)[1])

    df["Q7"] = df["Q7"].apply(challenge_basic.to_numeric)
    df["Q8"] = df["Q8"].apply(challenge_basic.to_numeric)
    df["Q9"] = df["Q9"].apply(challenge_basic.to_numeric)

    combined_condition = (
        (df['Q7'] >= -50) & (df['Q7'] <= 50) &
        (df['Q8'] >= 1) & (df['Q8'] <= 15) &
        (df['Q9'] >= 1) & (df['Q9'] <= 15)
    )

    df = df[combined_condition]

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
    cities = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']
    city_to_number = {city: i for i, city in enumerate(cities)}
    df['Label'] = df['Label'].map(city_to_number)
    # df['Label'] = pd.Categorical(df['Label'], categories=cities)
    # Label_onehot = pd.get_dummies(df['Label'], prefix='Label', dtype=int)
    
    df = pd.concat([df, Q1_onehot, Q2_onehot, Q3_onehot, Q4_onehot, Q6_max_onehot, Q6_min_onehot], axis=1)

    df['Q7'] = (df['Q7'] - df['Q7'].mean()) / (df['Q7'].std() + 0.0001)
    df['Q8'] = (df['Q8'] - df['Q8'].mean()) / (df['Q8'].std() + 0.0001)
    df['Q9'] = (df['Q9'] - df['Q9'].mean()) / (df['Q9'].std() + 0.0001)

    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'id', 'Q5', 'Q6', 'Q10', 'Q6_max', 'Q6_min'] # Edit Accordingly
    for col in delete_columns:
        del df[col]

    # df = df.sample(frac=1, random_state=100)
    
    # df.to_csv("UnNormalizedData.csv", index=False)
    df.to_csv("NormalizedData.csv", index=False)
