import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import re
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import seaborn as sns

file2 = 'nor_num.csv'
file3 = 'nor_oneH.csv'
num_labels = 4
wordsList = []
with open("words.txt", 'r') as file:
    wordsList = [line.strip() for line in file.readlines()]

def split_dataset(df, val_size, test_size):
    df_shuffled = df.sample(frac=1, random_state=42)
    X = df_shuffled.drop('Label', axis=1).values
    y = df_shuffled['Label'].values

    train_size = len(df) - val_size - test_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test

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


def cosine_similarity(X, v):
    c = np.zeros(X.shape[0], None, order='F')
    for i in range(X.shape[0]):
        dot_product = np.dot(X[i,:], v)
        norm_1 = np.linalg.norm(X[i,:])
        norm_2 = np.linalg.norm(v)
        cosine_sim = dot_product / (norm_1 * norm_2)
        c[i] = cosine_sim
    return c

def predict_knn(v, X_train, y_train, k):
    dists = cosine_similarity(X_train, v)
    indices = np.argsort(dists)[::-1][:k]
    ts = y_train[np.array(indices)]
    count_pair = {}
    for dist in ts:
      if dist in count_pair:
        count_pair[dist] += 1
      else:
        count_pair[dist] = 1
    prediction = max(count_pair, key=lambda k: count_pair[k])
    return prediction

def compute_accuracy(X_new, y_new, X_train, y_train, k):
    num_predictions = 0
    num_correct = 0
    for i in range(X_new.shape[0]): 
        v = X_new[i]
        t = y_new[i]
        y = predict_knn(v, X_train, y_train, k=k)
        if t == y:
           num_correct += 1
        num_predictions += 1
    return num_correct / num_predictions

if __name__ == "__main__":
    df = pd.read_csv(file2)
    df['Q10'].fillna("", inplace=True)
    # del df['Q10']
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df, val_size=200, test_size=200)
    vocab = get_vocab(X_train)

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

    # Using sklearn to make a knn model
    # 1. Euclidean distance metric
    e_train_acc = []
    e_valid_acc = []
    e_test_acc = []
    for k in range(1, 30):
        clf_e = KNeighborsClassifier(k)
        clf_e.fit(X_train, y_train)
        e_train_acc.append(clf_e.score(X_train, y_train))
        e_valid_acc.append(clf_e.score(X_val, y_val))
        e_test_acc.append(clf_e.score(X_test, y_test))
    k = np.argmax(e_valid_acc) + 1
    y_pred = clf_e.predict(X_test)
    cm_e = confusion_matrix(y_test, y_pred)
    print(f"(Euclidean confusion matrix): \n{cm_e}")

    # (Euclidean)plot confusion matrix 
    plt1.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(cm_e, annot=True, annot_kws={"size": 16}, fmt='g')  
    plt1.xlabel('Predicted labels')
    plt1.ylabel('True labels')
    plt1.title('Confusion Matrix')
    plt1.savefig("Euclidean confusion matrix.png")

    opt_k = np.argmax(e_valid_acc) + 1
    clf_e = KNeighborsClassifier(n_neighbors=opt_k)
    clf_e.fit(X_train, y_train)
    test_acc_e = clf_e.score(X_test, y_test)
    print(f"(Euclidean)Test Accuracy of the Optimistic k: {test_acc_e}")
    
    # 2. Manhattan distance metric
    m_train_acc = []
    m_valid_acc = []
    m_test_acc  = []
    for k in range(1, 30):
        clf_m = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        clf_m.fit(X_train, y_train)
        m_train_acc.append(clf_m.score(X_train, y_train))
        m_valid_acc.append(clf_m.score(X_val, y_val))
        m_test_acc.append(clf_m.score(X_test, y_test))
    y_pred = clf_m.predict(X_test)
    cm_m = confusion_matrix(y_test, y_pred)
    print(f"(Manhattan confusion matrix): \n{cm_m}")

    # (Manhattan)plot confusion matrix 
    plt2.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(cm_m, annot=True, annot_kws={"size": 16}, fmt='g')  
    plt2.xlabel('Predicted labels')
    plt2.ylabel('True labels')
    plt2.title('Confusion Matrix')
    plt2.savefig("Manhattan confusion matrix.png")

    opt_k = np.argmax(m_valid_acc) + 1
    clf_m = KNeighborsClassifier(n_neighbors=opt_k, metric='manhattan')
    clf_m.fit(X_train, y_train)
    test_acc_m = clf_m.score(X_test, y_test)
    print(f"(Manhattan)Test Accuracy of the Optimistic k: {test_acc_m}")
    
    # 3. Cosine similarity distance metric
    c_test_acc = []
    c_valid_acc = []
    c_test_acc  = []
    for k in range(1, 30):
        acc_val = compute_accuracy(X_val, y_val,X_train, y_train, k = k)
        acc_test = compute_accuracy(X_test, y_test,X_train, y_train, k = k)
        c_valid_acc.append(acc_val)
        c_test_acc.append(acc_test)
    y_pred = np.zeros(y_test.shape[0], None, order='F')
    for i in range(X_test.shape[0]): 
        v = X_test[i]
        t = y_test[i]
        y_pred[i:] = predict_knn(v, X_train, y_train, k=k)
    cm_c = confusion_matrix(y_test, y_pred)
    print(f"(Cosine similarity confusion matrix): \n{cm_c}")
    opt_k = np.argmax(c_valid_acc) + 1
    test_acc_c = compute_accuracy(X_test, y_test, X_train, y_train, opt_k)
    print(f"(Cosine similarity)Test Accuracy of the Optimistic k: {test_acc_c}")

    # (Cosine similarity)plot confusion matrix
    plt3.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(cm_c, annot=True, annot_kws={"size": 16}, fmt='g')  
    plt3.xlabel('Predicted labels')
    plt3.ylabel('True labels')
    plt3.title('Confusion Matrix')
    plt3.savefig("Cosine Similarity Confusion Matrix.png")

    # making the validation accuracy plots
    k_values = list(range(1, 30))
    plt.figure(figsize=(10, 6))
    plt.title("Validatation Accuracy for a kNN model")
    plt.plot(k_values, e_valid_acc, label='Euclidean distance')
    plt.plot(k_values, m_valid_acc, label='Manhattan distance')
    plt.plot(k_values, c_valid_acc, label='Cosine similarity')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("Knn Validation Accuracy.png")
    # making the test accuracy plots
    plt4.figure(figsize=(10, 6))
    plt4.title("Test Accuracy for a kNN model")
    plt4.plot(k_values, e_test_acc, label='Euclidean Test Accuracy')
    plt4.plot(k_values, m_test_acc, label='Manhattan Test Accuracy')
    plt4.plot(k_values, c_test_acc, label=' Cosine similarity Test Accuracy')
    plt4.title('KNN Test Accuracy vs. k Value')
    plt4.xlabel('k Value')
    plt4.ylabel('Accuracy')
    plt4.legend()
    plt4.grid(True)
    plt4.savefig("Test Accuracy.png")