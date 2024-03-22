import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ModifyData
import random

file_name = "ModifiedData.csv"

# file_name = "clean_dataset.csv"
# file_name = "pre_data.csv"
# From lab06
def make_onehot(indicies):
    I = np.eye(4)
    return I[indicies]

# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(z):
    return np.exp((z - np.max(z))) / np.sum(np.exp(z - np.max(z)))

# From lab06
class MLPModel(object):
    def __init__(self, num_features, num_hidden, num_classes):
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.W1 = np.zeros([num_features, num_hidden])
        self.b1 = np.zeros([num_hidden])
        self.W2 = np.zeros([num_hidden, num_classes])
        self.b2 = np.zeros([num_classes])
        self.initializeParams()
        self.cleanup()

    def initializeParams(self):
        self.W1 = np.random.normal(0, 2/self.num_features, self.W1.shape)
        self.b1 = np.random.normal(0, 2/self.num_features, self.b1.shape)
        self.W2 = np.random.normal(0, 2/self.num_hidden, self.W2.shape)
        self.b2 = np.random.normal(0, 2/self.num_hidden, self.b2.shape)

    def forward(self, X):
        self.N = X.shape[0]
        self.X = X
        self.m = np.dot(self.X, self.W1) + self.b1
        # From https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
        self.h = np.maximum(self.m, 0)
        self.z = np.dot(self.h, self.W2) + self.b2
        self.y = softmax(self.z)
        return self.y

    def backward(self, ts):
        self.z_bar = (self.y - ts) / self.N
        self.W2_bar = np.dot(self.h.T, self.z_bar)
        self.b2_bar = np.sum(self.z_bar, axis=0)
        self.h_bar = np.dot(self.z_bar, self.W2.T)
        # From https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
        self.m_bar = self.h_bar * (self.m > 0)
        self.W1_bar = np.dot(self.X.T, self.m_bar)
        self.b1_bar = np.sum(self.m_bar, axis=0)

    def loss(self, ts):
        return np.sum(-ts * np.log(self.y)) / ts.shape[0]

    def update(self, alpha):
        self.W1 = self.W1 - alpha * self.W1_bar
        self.b1 = self.b1 - alpha * self.b1_bar
        self.W2 = self.W2 - alpha * self.W2_bar
        self.b2 = self.b2 - alpha * self.b2_bar

    def cleanup(self):
        self.N = None
        self.X = None
        self.m = None
        self.h = None
        self.z = None
        self.y = None
        self.z_bar = None
        self.W2_bar = None
        self.b2_bar = None
        self.h_bar = None
        self.m_bar = None
        self.W1_bar = None
        self.b1_bar = None

def train_sgd(model, X_train, t_train, X_valid, t_valid, alpha=0.001, n_epochs=500, batch_size=100):
    train_loss = []
    valid_loss = []
    niter = 0
    N = X_train.shape[0]
    indices = list(range(N))

    for e in range(n_epochs):
        random.shuffle(indices)

        for i in range(0, N, batch_size):
            if (i + batch_size) > N:
                continue

            indices_in_batch = indices[i: i+batch_size]
            X_minibatch = X_train[indices_in_batch, :]
            t_minibatch = make_onehot(t_train[indices_in_batch])

            model.cleanup()
            model.forward(X_minibatch)
            model.backward(t_minibatch)
            model.update(alpha)

            train_loss.append(model.loss(t_minibatch))
            niter += 1

        model.cleanup()
        model.forward(X_valid)
        valid_loss.append((niter, model.loss(make_onehot(t_valid))))

    plt.title("SGD Training Curve Showing Loss at each Iteration")
    plt.plot(train_loss, label="Training Loss")
    plt.plot([iter for (iter, loss) in valid_loss], [loss for (iter, loss) in valid_loss], label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Validation.png")
    # print(train_loss, valid_loss)
    print("Final Training Loss:", train_loss[-1])
    print("Final Validation Loss:", valid_loss[-1])

if __name__ == "__main__":
    df = pd.read_csv(file_name).dropna()
    print(df.shape)
    X_train, t_train, X_valid, t_valid, X_test, t_test = ModifyData.split_dataset(df=df, val_size=200, test_size=216)
    print(X_train.shape, X_valid.shape, X_test.shape)
    num_features = X_train.shape[1]
    num_hidden = 100
    num_class = 4

    model = MLPModel(num_features=num_features, num_hidden=num_hidden, num_classes=num_class)
    train_sgd(model, X_train, t_train, X_valid, t_valid)
    print(model.y)

    