import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

file_name = "nor_oneH.csv"
log = open("AccuracyLog.txt", 'w')
parameterslog = open("ParametersLog.txt", 'w')
vocabList = open("Vocabulary.txt", "w", encoding="utf-8")
num_labels = 4
wordsList = []
with open("words.txt", 'r') as file:
    wordsList = [line.strip() for line in file.readlines()]

def split_dataset(df: pd.DataFrame, val_size: int, test_size: int):
    df_shuffled = df.sample(frac=1, random_state=42)
    X = df_shuffled.drop(columns='Label').values
    t = df_shuffled['Label'].values
    
    total_size = len(df_shuffled)
    train_size = total_size - (val_size + test_size)
    
    X_train, t_train = X[:train_size], t[:train_size]
    X_valid, t_valid = X[train_size:train_size + val_size], t[train_size:train_size + val_size]
    X_test, t_test = X[-test_size:], t[-test_size:]
    
    return X_train, t_train, X_valid, t_valid, X_test, t_test

data = pd.read_csv(file_name)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(data, 150, 150)

def get_vocab(X_train):
    vocab = []
    pattern = r"[^\w\s]"
    for i in range(X_train.shape[0]):
        text = re.sub(pattern, " ", X_train[i, 3])
        words = text.lower().split()
        words = [word.strip() for word in words]
        for word in words:
            if word in wordsList and word not in vocab:
                vocab.append(word)
    return sorted(vocab)

def insert_feature(data, vocab):
    features = np.zeros((data.shape[0], len(vocab)), dtype=float)
    for i in range(data.shape[0]):
        text = data[i, 3]
        words = set(re.sub(r"[^\w\s]", " ", text).lower().split())
        for word in words:
            if word in vocab:
                features[i, vocab.index(word)] = 1.0
    return features

vocab = get_vocab(X_train)
vocabList.write(str(vocab))
features = insert_feature(X_train, vocab)
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

input_size = X_train.shape[1]
hidden_size = 50
output_size = num_labels

model = TwoLayerNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
test_accuracies = []

num_epochs = 200
for epoch in range(num_epochs+1):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    _, predicted = torch.max(outputs.data, 1)
    train_total = y_train_tensor.size(0)
    train_correct = (predicted == y_train_tensor).sum().item()
    train_losses.append(train_loss)
    train_accuracies.append(100 * train_correct / train_total)
    if epoch % 10 == 0:
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.2f}%\n')

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        outputs = model(X_val_tensor)
        loss = criterion(outputs, y_val_tensor)
        val_loss = loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_total = y_val_tensor.size(0)
        val_correct = (predicted == y_val_tensor).sum().item()
    val_losses.append(val_loss)
    val_accuracies.append(100 * val_correct / val_total)
    if epoch % 10 == 0:
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {loss:.4f}, Validation Accuracy: {100 * val_correct / val_total:.2f}%\n')
    
    test_correct, test_total = 0, 0
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        test_total = y_test_tensor.size(0)
        test_correct = (predicted == y_test_tensor).sum().item()
    test_accuracies.append(100 * test_correct / test_total)
    if epoch % 10 == 0:
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {100 * test_correct / test_total:.2f}%\n')


weights_first_layer = model.layer1.weight.data
bias_first_layer = model.layer1.bias.data
weights_sec_layer = model.layer2.weight.data
bias_sec_layer = model.layer2.bias.data

weights_first_layer_np = weights_first_layer.cpu().numpy()
bias_first_layer_np = bias_first_layer.cpu().numpy()
weights_sec_layer_np = weights_sec_layer.cpu().numpy()
bias_sec_layer_np = bias_sec_layer.cpu().numpy()

np.savetxt('weights_first_layer.txt', weights_first_layer_np)
np.savetxt('bias_first_layer.txt', bias_first_layer_np)
np.savetxt('weights_second_layer.txt', weights_sec_layer_np)
np.savetxt('bias_second_layer.txt', bias_sec_layer_np)

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