from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import challenge_basic
import re

file_name = "clean_dataset.csv"
# Load your dataset
df = pd.read_csv("clean_dataset.csv")  # Make sure to use the correct path to your CSV file

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


def process_data(filename: str) -> pd.DataFrame:
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

    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'id', 'Q5', 'Q6', 'Q6_max', 'Q6_min'] # Edit Accordingly
    for col in delete_columns:
        del df[col]
        
    return df

df = process_data(df)
# Assuming 'Label' is your target variable
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vocab = []
# Assuming the rest of your code is unchanged...

# Make sure get_vocab and insert_feature functions work with DataFrames correctly
def get_vocab(X_train):
    vocab = set()
    pattern = r"[^\w\s]"
    # Adjust for DataFrame access
    for text in X_train.iloc[:, 3]:  # Assuming the 4th column contains text
        cleaned_text = re.sub(pattern, " ", text)
        words = cleaned_text.lower().split()
        vocab.update(words)
    return sorted(vocab)

def insert_feature(df, vocab):
    # Extract and clean the text column
    texts = df.iloc[:, 3].apply(lambda x: set(re.sub(r"[^\w\s]", " ", x).lower().split()))
    features = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, words in enumerate(texts):
        for j, word in enumerate(vocab):
            if word in words:
                features[i, j] = 1.0
    return features

# Apply feature extraction
vocab = get_vocab(X_train)
X_train_features = insert_feature(X_train, vocab)
X_test_features = insert_feature(X_test, vocab)

# Combine the original numeric data (excluding the text column) with the new features
X_train_combined = np.hstack([X_train.drop(X_train.columns[3], axis=1).values.astype(np.float64), X_train_features])
X_test_combined = np.hstack([X_test.drop(X_test.columns[3], axis=1).values.astype(np.float64), X_test_features])

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# The rest of your sklearn model definition, training, and evaluation remains the same...


# Define and train the model
# Adjust hyperparameters as needed. Here's a starting point based on your custom model
mlp = MLPClassifier(hidden_layer_sizes=(150,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=0.01)

mlp.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")

