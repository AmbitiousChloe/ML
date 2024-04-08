from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import re
from sklearn.model_selection import validation_curve

# Load your dataset
file_name = "nor_oneH.csv"
df = pd.read_csv(file_name)
X = df.drop('Label', axis=1)
y = df['Label']

wordsList=[]
with open("words.txt", 'r') as file:
    wordsList = [line.lower().strip() for line in file.readlines()]
    
# Functions for feature extraction remain unchanged
def get_vocab(X_train):
    vocab = set()
    pattern = r"[^\w\s]"
    texts = X_train.iloc[:, 3].fillna("").astype(str)
    for text in texts:
        cleaned_text = re.sub(pattern, " ", text)
        words = cleaned_text.lower().split()
        vocab.update(word for word in words if word in wordsList)
    return vocab

def insert_feature(df, vocab):
    texts = df.iloc[:, 3].fillna("").astype(str).apply(lambda x: set(re.sub(r"[^\w\s]", " ", x).lower().split()))
    features = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, words in enumerate(texts):
        for j, word in enumerate(vocab):
            if word in words:
                features[i, j] = 1.0
    return features

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
test_acc = []

for train_index, test_index in kf.split(X):
    # Splitting the dataset into the current train and test fold
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    X_train_target_fold, X_test_target_fold = y.iloc[train_index], y.iloc[test_index]
    
    
    # Feature extraction for the current fold
    vocab = get_vocab(X_train_fold)
    print(len(vocab))
    
    X_train_features = insert_feature(X_train_fold, vocab)
    X_test_features = insert_feature(X_test_fold, vocab)
    
    np.savetxt("X_test_features.txt", X_test_features, fmt="%s")
    
    # Combine with numeric data, excluding the text column
    X_train_combined = np.hstack([X_train_fold.drop(X_train_fold.columns[3], axis=1).values, X_train_features])
    X_test_combined = np.hstack([X_test_fold.drop(X_test_fold.columns[3], axis=1).values, X_test_features])
    
    np.savetxt("X_test_combined.txt", X_test_combined, fmt="%s")
    
    # Define and train the model
    mlp = MLPClassifier(hidden_layer_sizes=(150), max_iter=500, alpha=1e-0,
                        solver='sgd', learning_rate_init=0.01, random_state=42)
    mlp.fit(X_train_combined, X_train_target_fold)
    
        # Extract weights and biases
    weights_first_layer = mlp.coefs_[0]
    biases_first_layer = mlp.intercepts_[0]

    # For models with more than one layer, extract second layer weights and biases
    if len(mlp.coefs_) > 1:
        weights_second_layer = mlp.coefs_[1]
        biases_second_layer = mlp.intercepts_[1]

    # Save to text files
    np.savetxt("../NN_weights_1st_layer.txt", weights_first_layer, fmt="%s")
    np.savetxt("../NN_biases_1st_layer.txt", biases_first_layer, fmt="%s")

    if len(mlp.coefs_) > 1:
        np.savetxt("../NN_weights_2nd_layer.txt", weights_second_layer, fmt="%s")
        np.savetxt("../NN_biases_2nd_layer.txt", biases_second_layer, fmt="%s")
    
    # Make predictions and evaluate
    y_pred_fold = mlp.predict(X_test_combined)
    fold_accuracy = accuracy_score(X_test_target_fold, y_pred_fold)
    accuracies.append(fold_accuracy)
    
    test_accuracy = mlp.predict(X_train_combined)
    train_accuracy = accuracy_score(X_train_target_fold, test_accuracy)
    test_acc.append(train_accuracy)

# Calculate the average accuracy across all folds
average_accuracy = np.mean(accuracies)
print(accuracies)

print(f"Average K-Fold Accuracy: {average_accuracy:.4f}")
print(test_acc)
