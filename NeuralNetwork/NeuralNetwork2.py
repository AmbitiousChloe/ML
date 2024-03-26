import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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
        print(f"Epoch {epoch+1}: Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")


# Load and preprocess your dataset
df = pd.read_csv("ModifiedData.csv")  # Adjust the path to your dataset

# Split the dataset
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df, val_size=200, test_size=200)

# Instantiate and train the model
num_features = X_train.shape[1]
model = MLPModel(num_features, num_hidden=100, num_classes=4)  # Adjust parameters as necessary
train(model, X_train, y_train, X_val, y_val, lr=0.001, epochs=400)

# Test the model
y_pred_test = np.argmax(model.forward(X_test, training=False), axis=1)
test_accuracy = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

