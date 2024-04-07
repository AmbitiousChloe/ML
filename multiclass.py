"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

file_name = "clean_dataset.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Clean numerics

    df["Q7"] = df["Q7"].apply(to_numeric).fillna(0)

    # Clean for number categories

    df["Q1"] = df["Q1"].apply(get_number)

    # Create area rank categories
    
    """
    Q6
    "Skyscrapers=>2,Sport=>5,Art and Music=>1,Carnival=>6,Cuisine=>4,Economic=>3"
    
    rank_1  rank_2  rank_3  rank_4  rank_5  rank_6
       3       1       6       5       4       2
    """

    df["Q6"] = df["Q6"].apply(get_number_list_clean)

    temp_names = []
    for i in range(1,7):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))

    del df["Q6"]

    # Create category indicators

    new_names = []
    for col in ["Q1"] + temp_names:
        indicators = pd.get_dummies(df[col], prefix=col)
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]

    # Create multi-category indicators

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    del df["Q5"]

    # Prepare data for training - use a simple train/test split for now

    df = df[new_names + ["Q7", "Label"]]

    df = df.sample(frac=1, random_state=random_state)

    x = df.drop("Label", axis=1).values
    y = pd.get_dummies(df["Label"].values)

    # n_train = 1200

    # x_train = x[:n_train]
    # y_train = y[:n_train]

    # x_test = x[n_train:]
    # y_test = y[n_train:]

    Y = df['Label']  # Target
    X = df.drop(['Label'], axis=1)  # Features
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)   
    # Train and evaluate classifiers

    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
    
    # Train the model
    log_reg.fit(X_train, Y_train)

    # Predictions
    Y_pred = log_reg.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

