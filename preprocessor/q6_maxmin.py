import random
import pandas as pd
import numpy as np
import re

file_name = "clean_dataset.csv"



"""###############################
Helper functions for reading data:
"""
def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]


def get_number(s, na):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else na


def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0


def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)



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
      value = random.choice(['3', '4'])
    # Convert value to integer and add to the dictionary
    result_dict[key.strip()] = int(value.strip())
    result_dict = radm_dict(result_dict)
  return [max(result_dict, key=result_dict.get), min(result_dict, key=result_dict.get)]


def process_data(filename: str) -> pd.DataFrame:
    # read the data as pandas dataframe
    df = pd.read_csv(file_name)

    # pre process Q1 - Q4
    na_14 = 2
    df["Q1"] = df["Q1"].apply(lambda x: get_number(x, na_14))
    df["Q2"] = df["Q2"].apply(lambda x: get_number(x, na_14))
    df["Q3"] = df["Q3"].apply(lambda x: get_number(x, na_14))
    df["Q4"] = df["Q4"].apply(lambda x: get_number(x, na_14))

        # fill all the missing values in Q1-Q4 colums as 2
    df["Q1"].fillna(na_14, inplace = True)
    df["Q2"].fillna(na_14, inplace = True)
    df["Q3"].fillna(na_14, inplace = True)
    df["Q4"].fillna(na_14, inplace = True)
    
    Q1_Q4_all_categories = [1, 2, 3, 4, 5]

    for col in ['Q1', 'Q2', 'Q3', 'Q4']:
        df[col] = pd.Categorical(df[col], categories=Q1_Q4_all_categories)

    Q1_onehot = pd.get_dummies(df['Q1'], prefix='Q1', dtype=int)
    Q2_onehot = pd.get_dummies(df['Q2'], prefix='Q2', dtype=int)
    Q3_onehot = pd.get_dummies(df['Q3'], prefix='Q3', dtype=int)
    Q4_onehot = pd.get_dummies(df['Q4'], prefix='Q4', dtype=int)
   
    # pre process Q5
    q5_category = ["Partner", "Friends", "Siblings", "Co-worker"]
   
    # fill all missing values in Q5 by randomly choose one of the possible categories
    df["Q5"].fillna(random.choice(q5_category))

    # create one hot features for 4 categories
    for cat in q5_category:
        cat_name = f"{cat}"
        df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

        # pre process Q6 as NUMERICAL
        # fill all the missing values in Q6 as mean (3)
    df["Q6"].fillna("Skyscrapers=>3,Sport=>3,Art and Music=>3,Carnival=>3,Cuisine=>3,Economic=>3", 
                        inplace = True)
        # Q6 max min
    df['Q6_max'] = df['Q6'].apply(lambda x: to_dict(x)[0])
    df['Q6_min'] = df['Q6'].apply(lambda x: to_dict(x)[1])

        # Onehot Encoding
    Q6_categories = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']
    df['Q6_max'] = pd.Categorical(df['Q6_max'], categories=Q6_categories)
    Q6_max_onehot = pd.get_dummies(df['Q6_max'], prefix='Q6_max', dtype=int)
    df['Q6_min'] = pd.Categorical(df['Q6_min'], categories=Q6_categories)
    Q6_min_onehot = pd.get_dummies(df['Q6_min'], prefix='Q6_min', dtype=int)
    
        # Q6 num
        # df["Q6"] = df["Q6"].apply(lambda x: to_dict_complete(x))sx
    
    # pre process Q7 - Q9
    df["Q7"] = df["Q7"].apply(to_numeric)
    df["Q8"] = df["Q8"].apply(to_numeric)
    df["Q9"] = df["Q9"].apply(to_numeric)

        # fill all the missing values in Q7 - Q9 with column mean
    df["Q7"].fillna(df["Q7"].mean(), inplace = True)
    df["Q8"].fillna(df["Q8"].mean(), inplace = True)
    df["Q9"].fillna(df["Q9"].mean(), inplace = True)

        # replace all the outliers by given number
    q7_min = -30
    q7_max = 45
    q89_min = 1
    Q89_max = 15
    df.loc[(df['Q7'] < q7_min), 'Q7'] = q7_min
    df.loc[(df['Q7'] > q7_max), 'Q7'] = q7_max
    df.loc[(df['Q8'] < q89_min), 'Q8'] = q89_min
    df.loc[(df['Q9'] > Q89_max), 'Q9'] = Q89_max
    df.loc[(df['Q8'] < q89_min), 'Q8'] = q89_min
    df.loc[(df['Q9'] > Q89_max), 'Q9'] = Q89_max
    
        # normalizing
    df['Q7'] = (df['Q7'] - df['Q7'].mean()) / (df['Q7'].std() + 0.0001)
    df['Q8'] = (df['Q8'] - df['Q8'].mean()) / (df['Q8'].std() + 0.0001)
    df['Q9'] = (df['Q9'] - df['Q9'].mean()) / (df['Q9'].std() + 0.0001)

    # fixing na in Q10
    df['Q10'] = df['Q10'].fillna(" ").astype(str)

        # map labels to numbers
    cities = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']
    city_to_number = {city: i for i, city in enumerate(cities)}
    df['Label'] = df['Label'].map(city_to_number)
        
        # add extra colums to df
    df = pd.concat([df, Q1_onehot, Q2_onehot, Q3_onehot, Q4_onehot, Q6_max_onehot, Q6_min_onehot], axis=1)

        # delete non useful columns
    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'id', 'Q5', 'Q6', 'Q6_max', 'Q6_min'] # Edit Accordingly
    for col in delete_columns:
        del df[col]

    return df


if __name__ == "__main__":
    df = process_data(file_name)
    df_ori = pd.read_csv(file_name)
    df.to_csv("nor_maxmin.csv", index=False)
    print(df.columns[df.isna().any()].tolist())