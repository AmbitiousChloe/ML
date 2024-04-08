import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Load your dataset
df = pd.read_csv("NormalizedData.csv")  # Update the path to your CSV file

X = df.drop('Label', axis=1).values
y = df['Label'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=150, shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super(NeuralNet, self).__init__()
        self.layer_1 = nn.Linear(num_features, 70)  # 70 neurons in the hidden layer
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(70, len(pd.unique(y)))  # Output layer
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

model = NeuralNet(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:  # Print every 100th epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    y_pred_train = torch.argmax(y_pred_train, dim=1)
    train_accuracy = accuracy_score(y_train_tensor, y_pred_train)

    y_pred_test = model(X_test_tensor)
    y_pred_test = torch.argmax(y_pred_test, dim=1)
    test_accuracy = accuracy_score(y_test_tensor, y_pred_test)

print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
