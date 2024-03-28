import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import challenge_basic
import re

file_name = "clean_dataset.csv"

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

def softmax(z):
    
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def make_onehot(indices):
    I = np.eye(4)  # Adjust based on your dataset's number of classes
    return I[indices]

class MLPModel(object):
    def __init__(self, num_features, num_hidden, num_classes):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # He initialization for weights
        self.W1 = np.random.randn(num_features, num_hidden) * np.sqrt(2. / num_features)
        self.b1 = np.zeros(num_hidden)
        self.W2 = np.random.randn(num_hidden, num_classes) * np.sqrt(2. / num_hidden)
        self.b2 = np.zeros(num_classes)

    def forward(self, X, training=True):
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        # Output layer error
        error = self.a2 - y_true
        self.W2_bar = np.dot(self.a1.T, error)
        self.b2_bar = np.sum(error, axis=0)

        # Hidden layer error
        error_hidden = np.dot(error, self.W2.T) * (self.z1 > 0)
        self.W1_bar = np.dot(X.T, error_hidden)
        self.b1_bar = np.sum(error_hidden, axis=0)

    def update(self, lr):
        self.W1 -= lr * self.W1_bar
        self.b1 -= lr * self.b1_bar
        self.W2 -= lr * self.W2_bar
        self.b2 -= lr * self.b2_bar

def split_dataset(df, val_size, test_size):
    df_shuffled = df.sample(frac=1, random_state=42)
    X = df_shuffled.drop('Label', axis=1).values
    y = df_shuffled['Label'].values

    train_size = len(df) - val_size - test_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def train(model, X_train, y_train, X_val, y_val, lr=0.001, epochs=100):
    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs):
        # Using the entire dataset as one batch for Gradient Descent
        y_train_onehot = make_onehot(y_train)
        model.forward(X_train, training=True)
        model.backward(X_train, y_train_onehot)
        model.update(lr)
        
        # Monitor training and validation accuracy
        train_predictions = np.argmax(model.forward(X_train, training=False), axis=1)
        train_accuracy = np.mean(train_predictions == y_train)
        val_predictions = np.argmax(model.forward(X_val, training=False), axis=1)
        val_accuracy = np.mean(val_predictions == y_val)
        
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return train_accuracies, val_accuracies
        

# Load and preprocess your dataset
df = process_data(file_name)  # Adjust the path to your dataset

# Split the dataset
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df, val_size=200, test_size=200)

vocab = []
def get_vocab(X_train):
    for i in range(X_train.shape[0]):
        pattern = r"[.?,;:-]"
        q = re.sub(pattern, " ", X_train[i,3])
        lst = q.split()
        X_train[i,3] = lst
        for w in lst:
            if w not in vocab:
                vocab.append(w)
    return vocab

vocab = get_vocab(X_train)
def insert_feature(nparray, vocab):   
    wl = nparray[:, 3]    
    x = np.zeros((nparray.shape[0], len(vocab)), dtype=float)
    for i in range(nparray.shape[0]):
        for j in range(len(vocab)):
            if vocab[j] in wl[i]:
                x[i, j] = 1.0
    return x

features = insert_feature(X_train,  vocab)
X_train_numeric = np.delete(X_train, 3, axis=1).astype(np.float64)
X_train = np.hstack((X_train_numeric, features)).astype(np.float64)
X_train = np.concatenate((X_train[:, :3], X_train[:, 4:]), axis=1)

valid_features = insert_feature(X_val, vocab)
X_val_numeric = np.delete(X_val, 3, axis=1).astype(np.float64)
X_val = np.hstack((X_val_numeric, valid_features)).astype(np.float64)
X_val = np.concatenate((X_val[:, :3], X_val[:, 4:]), axis=1)

test_features = insert_feature(X_test, vocab)
X_test_numeric = np.delete(X_test, 3, axis=1).astype(np.float64)
X_test = np.hstack((X_test_numeric, test_features)).astype(np.float64)
X_test = np.concatenate((X_test[:, :3], X_test[:, 4:]), axis=1)


# Instantiate and train the model
num_features = X_train.shape[1]

model = MLPModel(num_features, num_hidden=100, num_classes=4)  # Adjust parameters as necessary
train_accuracies, val_accuracies = train(model, X_train, y_train, X_val, y_val, lr=0.001, epochs=400)

# Test the model
y_pred_test = np.argmax(model.forward(X_test, training=False), axis=1)
test_accuracy = np.mean(y_pred_test == y_test)
print(y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# plt.figure(figsize=(12, 5))
# plt.plot(train_accuracies, label='Training Accuracy')
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()



