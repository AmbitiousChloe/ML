import re
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# file_name = "clean_dataset.csv"
# file_name = "pre_data.csv"
random_state = 42
file_name = "NormalizedData.csv"

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


if __name__ == "__main__":
    df = pd.read_csv(file_name).dropna()
   
    X_train, t_train, X_valid, t_valid, X_test, t_test = split_dataset(df=df, val_size=200, test_size=216)
    train_acc = []
    test_acc = []
    for k in range(1, 50):
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
    
    k_values = list(range(1, 50))  # Assuming k values range from 1 to 12 for demonstration

# Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title('KNN Accuracy vs. k Value')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("KNNgraph.png")