from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "../ModifiedData.csv"
log = open("AccuracyLog.txt", 'w')
parameterslog = open("ParametersLog.txt", 'w')
data = pd.read_csv(file_name).dropna()
num_labels = 4

X = data.drop('Label', axis=1).values
y = data['Label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
test = data.sample(frac=1, random_state=311)
x_test_set = test.drop("Label", axis=1).values
y_test_set = test["Label"].values
X_test = x_test_set[:150]
y_test = y_test_set[:150]

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
        log.write(f'Epoch [{epoch+1}/{num_epochs}], Test size: {test_total:.4f}, Test Accuracy: {100100 * test_correct / test_total:.2f}%\n')

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