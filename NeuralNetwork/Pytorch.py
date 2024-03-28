from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import challenge_basic
import matplotlib.pyplot as plt
import re

file_name = "clean_dataset.csv"
log = open("AccuracyLog.txt", 'w')
parameterslog = open("ParametersLog.txt", 'w')
num_labels = 4

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

    delete_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'id', 'Q5', 'Q6', 'Q6_max', 'Q6_min']
    for col in delete_columns:
        del df[col]
    return df

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

data = process_data(file_name)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(data, 150, 150)

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

X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=150, shuffle=False)

input_size = X_train.shape[1]
hidden_size = 20
output_size = num_labels

# Model, criterion, and optimizer initialization remains the same
model = TwoLayerNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to track the accuracy and loss over time for both training and validation sets
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
test_accuracies = []

num_epochs = 500
for epoch in range(num_epochs+1):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(100 * train_correct / train_total)
    if epoch % 10 == 0:
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.2f}%\n')
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100 * val_correct / val_total)
    if epoch % 10 == 0:
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {loss:.4f}, Validation Accuracy: {100 * val_correct / val_total:.2f}%\n')
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracies.append(100 * test_correct / test_total)
    if epoch % 10 == 0:
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Test size: {test_total:.4f}, Test Accuracy: {100 * test_correct / test_total:.2f}%\n')

weights_first_layer = model.layer1.weight.data
bias_first_layer = model.layer1.bias.data
weights_sec_layer = model.layer2.weight.data
bias_sec_layer = model.layer2.bias.data

parameterslog.write(f"Weights of the first layer:\n{weights_first_layer}\n")
parameterslog.write(f"Bias of the first layer:\n{bias_first_layer}\n\n\n")
parameterslog.write(f"Weights of the second layer:\n{weights_sec_layer}\n")
parameterslog.write(f"Bias of the second layer:\n{bias_sec_layer}\n")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("pytorch.png")