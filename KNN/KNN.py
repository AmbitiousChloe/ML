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
from sklearn.model_selection import KFold

file2 = 'nor_num.csv'
file3 = 'nor_oneH.csv'
num_labels = 4

wordsList=[]
with open("words.txt", 'r') as file:
    wordsList = [line.lower().strip() for line in file.readlines()]

def split_dataset(df, test_size):
    df_shuffled = df.sample(frac=1, random_state=42)
    X = df_shuffled.drop('Label', axis=1).values
    y = df_shuffled['Label'].values

    train_size = len(df) - test_size
    X_new, y_new = X[:train_size], y[:train_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    return X_new, y_new, X_test, y_test

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


def kfold_split(k: int, X: np.ndarray, Y: np.ndarray):
    nsample = X.shape[0] // k
    remain = X.shape[0] % k
    spl_X = []
    spl_Y = []
    start = 0
    
    for i in range(k):
        end = start + nsample + (1 if i < remain else 0)
        spl_X.append(X[start:end])
        spl_Y.append(Y[start:end])
        start = end
    return spl_X, spl_Y


def euc(xtrain: np.ndarray, ytrain: np.ndarray, xval, yval, k_range: int, pred = False):
    e_train_acc = []
    e_valid_acc = []
    for k in range(1, k_range):
        clf_e = KNeighborsClassifier(k)
        clf_e.fit(xtrain, ytrain)
        e_train_acc.append(clf_e.score(xtrain, ytrain))
        e_valid_acc.append(clf_e.score(xval, yval))
    best_k = np.argmax(e_valid_acc) + 1
    return best_k 


def man(xtrain: np.ndarray, ytrain: np.ndarray, xval, yval, k_range: int, pred = False):
    m_train_acc = []
    m_valid_acc  = []
    for k in range(1, k_range):
        clf_m = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        clf_m.fit(xtrain, ytrain)
        m_train_acc.append(clf_m.score(xtrain, ytrain))
        m_valid_acc.append(clf_m.score(xval, yval))
    best_k = np.argmax(m_valid_acc) + 1
    return best_k


# def cos(xtrain: np.ndarray, ytrain: np.ndarray, xval, yval, k_range: int, pred = False):
#     c_train_acc = []
#     c_valid_acc = []
#     for k in range(1, 30):
#         acc_val = compute_accuracy(xval, yval,xtrain, ytrain, k)
#         acc_train = compute_accuracy(xtrain, y_test,X_train, y_train, k)
#         c_valid_acc.append(acc_val)
#         c_test_acc.append(acc_test)


if __name__ == "__main__":
    df = pd.read_csv(file3)
    df['Q10'].fillna("", inplace=True)
    X_train, y_train, X_test, y_test = split_dataset(df, test_size=200)

    vocab = get_vocab(X_train)
    test_feat = insert_feature(X_test, vocab)

    X_test_numeric = np.delete(X_test, 3, axis=1).astype(np.float64)
    X_test = np.hstack((X_test_numeric, test_feat)).astype(np.float64)


    ########## K-Fold Setting
    k = 5
    ##########

    x_lst, y_lst = kfold_split(k, X_train, y_train)
    euc_k_log = {}
    man_k_log = {}
    cos_k_log = {}

    for i in range(k):
        remin_index = list(range(k))
        remin_index.remove(i)

        xval = x_lst[i]
        xtrain = np.concatenate([x_lst[i] for i in remin_index], axis = 0)

        vocab = get_vocab(xtrain)
        train_feature = insert_feature(xtrain, vocab)
        val_feature = insert_feature(xval, vocab)

        xtrain_numeric = np.delete(xtrain, 3, axis=1).astype(np.float64)
        xtrain = np.hstack((xtrain_numeric, train_feature)).astype(np.float64)

        xval_numeric = np.delete(xval, 3, axis=1).astype(np.float64)
        xval = np.hstack((xval, val_feature)).astype(np.float64)

        # xval = np.hstack([xval.drop(xval.columns[3], axis = 1).values.astype(np.float64), train_feature])

        yval = y_lst[i]
        ytrain = np.concatenate([y_lst[i] for i in remin_index], axis = 0)

        euc_k = euc(xtrain, ytrain, xval, yval, 25)
        man_k = man(xtrain, ytrain, xval, yval, 25)

        if euc_k in euc_k_log:
            euc_k_log[euc_k] += 1
        else:
            euc_k_log[euc_k] = 0

        if man_k in man_k_log:
            man_k_log[man_k] += 1
        else:
            man_k_log[man_k] = 1

    print(euc_k_log, man_k_log)


    # # 1. Euclidean distance metric
    # e_train_acc = []
    # e_valid_acc = []
    # e_test_acc = []
    # for k in range(1, 30):
    #     clf_e = KNeighborsClassifier(k)
    #     clf_e.fit(X_train, y_train)
    #     e_train_acc.append(clf_e.score(X_train, y_train))
    #     e_valid_acc.append(clf_e.score(X_val, y_val))
    #     e_test_acc.append(clf_e.score(X_test, y_test))
    # k = np.argmax(e_valid_acc) + 1
    # y_pred = clf_e.predict(X_test)
    # # confusion matrix of euclidean distance
    # cm_e = confusion_matrix(y_test, y_pred)
    # # precision of euclidean 
    # tp_e = cm_e[1, 1] # True Positives are on the diagonal, assuming binary classification for simplicity
    # fp_e = cm_e[0, 1] # False Positives are the off-diagonal in the same predicted column
    # precision_e = tp_e / (tp_e + fp_e)
    # print(f"(Euclidean) Precision: {precision_e:.2f}")
    # # (Euclidean)plot confusion matrix 
    # plt1.figure(figsize=(10,7))
    # sns.set(font_scale=1.4)  # for label size
    # sns.heatmap(cm_e, annot=True, annot_kws={"size": 16}, fmt='g')  
    # plt1.xlabel('Predicted labels')
    # plt1.ylabel('True labels')
    # plt1.title('(Euclidean)Confusion Matrix')
    # plt1.savefig("Euclidean confusion matrix.png")

    # opt_k = np.argmax(e_valid_acc) + 1
    # clf_e = KNeighborsClassifier(n_neighbors=opt_k)
    # clf_e.fit(X_train, y_train)
    # test_acc_e = clf_e.score(X_test, y_test)
    # print(f"(Euclidean)Test Accuracy of the Optimistic k: {test_acc_e}")
    
    # # 2. Manhattan distance metric
    # m_train_acc = []
    # m_valid_acc = []
    # m_test_acc  = []
    # for k in range(1, 30):
    #     clf_m = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    #     clf_m.fit(X_train, y_train)
    #     m_train_acc.append(clf_m.score(X_train, y_train))
    #     m_valid_acc.append(clf_m.score(X_val, y_val))
    #     m_test_acc.append(clf_m.score(X_test, y_test))
    # y_pred = clf_m.predict(X_test)
    # # confusion matrix of manhattan
    # cm_m = confusion_matrix(y_test, y_pred)
    # # precision of manhattan
    # tp_m = cm_m[1, 1]
    # fp_m = cm_m[0, 1]
    # precision_m = tp_m / (tp_m + fp_m)
    # print(f"(Manhattan) Precision: {precision_m:.2f}")

    # # (Manhattan)plot confusion matrix 
    # plt2.figure(figsize=(10,7))
    # sns.set(font_scale=1.4)  # for label size
    # sns.heatmap(cm_m, annot=True, annot_kws={"size": 16}, fmt='g')  
    # plt2.xlabel('Predicted labels')
    # plt2.ylabel('True labels')
    # plt2.title('(Manhattan)Confusion Matrix')
    # plt2.savefig("Manhattan confusion matrix.png")

    # opt_k = np.argmax(m_valid_acc) + 1
    # clf_m = KNeighborsClassifier(n_neighbors=opt_k, metric='manhattan')
    # clf_m.fit(X_train, y_train)
    # test_acc_m = clf_m.score(X_test, y_test)
    # print(f"(Manhattan)Test Accuracy of the Optimistic k: {test_acc_m}")
    
    # # 3. Cosine similarity distance metric
    # c_test_acc = []
    # c_valid_acc = []
    # c_test_acc  = []
    # for k in range(1, 30):
    #     acc_val = compute_accuracy(X_val, y_val,X_train, y_train, k = k)
    #     acc_test = compute_accuracy(X_test, y_test,X_train, y_train, k = k)
    #     c_valid_acc.append(acc_val)
    #     c_test_acc.append(acc_test)
    # y_pred = np.zeros(y_test.shape[0], None, order='F')
    # for i in range(X_test.shape[0]): 
    #     v = X_test[i]
    #     t = y_test[i]
    #     y_pred[i:] = predict_knn(v, X_train, y_train, k=k)
    # # confusion matrix of cosine similarity
    # cm_c = confusion_matrix(y_test, y_pred)
    # # precision of cosine similarity
    # tp_c = cm_c[1, 1]
    # fp_c = cm_c[0, 1]
    # precision_c = tp_c / (tp_c + fp_c)
    # print(f"(Cosine similarity) Precision: {precision_c:.2f}")

    # opt_k = np.argmax(c_valid_acc) + 1
    # test_acc_c = compute_accuracy(X_test, y_test, X_train, y_train, opt_k)
    # print(f"(Cosine similarity)Test Accuracy of the Optimistic k: {test_acc_c}")

    # # (Cosine similarity)plot confusion matrix
    # plt3.figure(figsize=(10,7))
    # sns.set(font_scale=1.4)  # for label size
    # sns.heatmap(cm_c, annot=True, annot_kws={"size": 16}, fmt='g')  
    # plt3.xlabel('Predicted labels')
    # plt3.ylabel('True labels')
    # plt3.title('(Cosine similarity) Confusion Matrix')
    # plt3.savefig("Cosine Similarity Confusion Matrix.png")

    # # making the validation accuracy plots
    # k_values = list(range(1, 30))
    # plt.figure(figsize=(10, 6))
    # plt.title("Validatation Accuracy for a kNN model")
    # plt.plot(k_values, e_valid_acc, label='Euclidean distance')
    # plt.plot(k_values, m_valid_acc, label='Manhattan distance')
    # plt.plot(k_values, c_valid_acc, label='Cosine similarity')
    # plt.xlabel("k")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig("Knn Validation Accuracy.png")
    # # making the test accuracy plots
    # plt4.figure(figsize=(10, 6))
    # plt4.title("Test Accuracy for a kNN model")
    # plt4.plot(k_values, e_test_acc, label='Euclidean Test Accuracy')
    # plt4.plot(k_values, m_test_acc, label='Manhattan Test Accuracy')
    # plt4.plot(k_values, c_test_acc, label=' Cosine similarity Test Accuracy')
    # plt4.xlabel('k Value')
    # plt4.ylabel('Accuracy')
    # plt4.legend()
    # plt4.grid(True)
    # plt4.savefig("Test Accuracy.png")