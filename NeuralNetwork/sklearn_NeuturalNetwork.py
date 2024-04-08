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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

wordsList=[]
with open("words.txt", 'r') as file:
    wordsList = [line.lower().strip() for line in file.readlines()]
    
file_name = "nor_oneH.csv"

# Load your dataset
df = pd.read_csv(file_name)
# Assuming 'Label' is your target variable
X = df.drop('Label', axis=1)
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Make sure get_vocab and insert_feature functions work with DataFrames correctly
def get_vocab(X_train):
    vocab = set()
    pattern = r"[^\w\s]"
    # Assuming the text is in the 4th column, adjust the index as necessary
    texts = X_train.iloc[:, 3].fillna("").astype(str)  # Handle NaN values and ensure string type
    for text in texts:
        cleaned_text = re.sub(pattern, " ", text)
        words = cleaned_text.lower().split()
        vocab.update(word for word in words if word in wordsList)
    return vocab


def insert_feature(df, vocab):
    # Extract and clean the text column, ensuring all entries are treated as strings
    texts = df.iloc[:, 3].fillna("").astype(str).apply(lambda x: set(re.sub(r"[^\w\s]", " ", x).lower().split()))
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

# Define and train the model
# Adjust hyperparameters as needed. Here's a starting point based on your custom model
mlp = MLPClassifier(hidden_layer_sizes=(150), max_iter=250, alpha=1e-4,
                    solver='sgd', verbose=True,
                    learning_rate_init=0.01)

mlp.fit(X_train_combined, y_train)

# Make predictions and evaluate the model
y_pred = mlp.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
y_train_pred = mlp.predict(X_train_combined)

# Calculate and print the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

