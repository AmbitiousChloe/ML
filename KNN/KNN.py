import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import re
import numpy as np
import random


# random_state = 42
# file_name = "NormalizedData.csv"
# file1 = 'nor_maxmin.csv'
file2 = 'nor_num.csv'
file3 = 'nor_oneH.csv'

def split_dataset_knn(df, test_size):
    df_shuffled = df.sample(frac=1, random_state=42)
    X = df_shuffled.drop('Label', axis=1).values
    y = df_shuffled['Label'].values

    train_size = len(df) - test_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    return X_train, y_train, X_test, y_test

def get_vocab(X_train):
    vocab = set()  # Use a set for efficiency in checking membership
    pattern = r"[^\w\s]"  # This pattern matches anything that's not alphanumeric or whitespace
    for i in range(X_train.shape[0]):
        text = re.sub(pattern, " ", X_train[i, 3])
        words = text.lower().split()
        words = [word.strip() for word in words]  # Strip whitespace
        vocab.update(words)  # Add cleaned words to the vocabulary
    return sorted(vocab)  # Convert to a sorted list before returning

def insert_feature(nparray, vocab):
    features = np.zeros((nparray.shape[0], len(vocab)), dtype=float)
    for i in range(nparray.shape[0]):
        text = nparray[i, 3]
        words = set(re.sub(r"[^\w\s]", " ", text).lower().split())
        for j, word in enumerate(vocab):
            if word in words:
                features[i, j] = 1.0
    return features

if __name__ == "__main__":
    df = pd.read_csv(file3)
    df['Q10'].fillna("", inplace=True)
    # del df['Q10']
    X_train, t_train, X_test, t_test = split_dataset_knn(df=df, test_size=216)
    vocab = get_vocab(X_train)

    features = insert_feature(X_train,  vocab)
    X_train_numeric = np.delete(X_train, 3, axis=1).astype(np.float64)
    X_train = np.hstack((X_train_numeric, features)).astype(np.float64)
    X_train = np.concatenate((X_train[:, :3], X_train[:, 4:]), axis=1)

    test_features = insert_feature(X_test, vocab)
    X_test_numeric = np.delete(X_test, 3, axis=1).astype(np.float64)
    X_test = np.hstack((X_test_numeric, test_features)).astype(np.float64)
    X_test = np.concatenate((X_test[:, :3], X_test[:, 4:]), axis=1)

    train_acc = []
    test_acc = []
    for k in range(1, 30):
        clf = KNeighborsClassifier(k)
        clf.fit(X_train, t_train)
        train_acc.append(clf.score(X_train, t_train))
        test_acc.append(clf.score(X_test, t_test))

    print(f"{type(clf).__name__} train acc: {train_acc}")
    print(f"{type(clf).__name__} test acc: {test_acc}")
    opt_k = test_acc.index(max(test_acc))
    opt_test_acc = max(test_acc)
    print(f"optimize test acc: {opt_test_acc}")
    print(f"optimize k: {opt_k + 1}")
    
    k_values = list(range(1, 30))  # Assuming k values range from 1 to 12 for demonstration

# Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_acc, label='Training Accuracy')
    plt.plot(k_values, test_acc, label='Test Accuracy')
    plt.title('KNN Accuracy vs. k Value')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=1)
    plt.xticks(k_values)
    plt.savefig("KNNgraph.png")