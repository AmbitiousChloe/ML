import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def pred_multiclass(w, X):
    """
    Compute the prediction made by a multiclass logistic regression model with weights `w`
    on the dataset with input data matrix `X`.

    Parameters:
        `w` - a numpy array of shape (D+1, C), where C is the number of classes
        `X` - data matrix of shape (N, D+1)

    Returns: Prediction matrix `y` of shape (N, C).
    """
    return sigmoid(np.dot(X, w))

def loss_multiclass(w, X, T):
    """
    Compute the average cross-entropy loss of a multiclass logistic regression model
    with weights `w` on the dataset with input data matrix `X` and one-hot encoded targets `T`.

    Parameters:
        `w` - a numpy array of shape (D+1, C)
        `X` - data matrix of shape (N, D+1)
        `T` - one-hot encoded target matrix of shape (N, C)

    Returns: scalar cross entropy loss value.
    """
    Y = pred_multiclass(w, X)
    return -np.mean(np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y), axis=1))

def grad_multiclass(w, X, T):
    """
    Compute the gradient of the loss function for multiclass logistic regression.

    Parameters:
        `w` - a numpy array of shape (D+1, C)
        `X` - data matrix of shape (N, D+1)
        `T` - one-hot encoded target matrix of shape (N, C)

    Returns: Gradient matrix of shape (D+1, C).
    """
    N = X.shape[0]
    Y = pred_multiclass(w, X)
    return np.dot(X.T, Y - T) / N

def accuracy_multiclass(w, X, t):
    """
    Compute the accuracy of a multiclass logistic regression model with weights `w`
    on the dataset with input data matrix `X` and target vector `t`.

    Parameters:
        `w` - a numpy array of shape (D+1, C)
        `X` - data matrix of shape (N, D+1)
        `t` - target vector of shape (N), with integer class labels

    Returns: accuracy value, between 0 and 1.
    """
    Y = pred_multiclass(w, X)
    predictions = np.argmax(Y, axis=1)
    return np.mean(predictions == t)



#Logistic Regression with Stochastic Gradient Descent
def solve_via_sgd_multiclass(alpha=0.0025, n_epochs=0, batch_size=100,
                             X_train=None, t_train=None,
                             X_valid=None, t_valid=None,
                             C=0, w_init=None, plot=True):
    """
    Given `alpha` - the learning rate
          `n_epochs` - the number of **epochs** of gradient descent to run
          `batch_size` - the size of each mini batch
          `X_train` - the data matrix to use for training
          `t_train` - the one-hot encoded target matrix to use for training
          `X_valid` - the data matrix to use for validation
          `t_valid` - the one-hot encoded target matrix to use for validation
          `C` - the number of classes
          `w_init` - the initial `w` matrix (if `None`, use a matrix of zeros)
          `plot` - whether to track statistics and plot the training curve

    Solves for logistic regression weights via stochastic gradient descent,
    using the provided batch size for multiclass classification.

    Returns weights after `n_epochs` iterations.
    """
    N, D = X_train.shape

    # Initialize weights
    if w_init is None:
        w = np.zeros((D, C))
    else:
        w = w_init

    # Tracking loss and accuracy
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    indices = np.arange(N)

    for epoch in range(n_epochs):
        np.random.shuffle(indices)

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            if end_idx <= start_idx:
                continue  # Skip this batch if it's smaller than desired batch size
            
            batch_indices = indices[start_idx:end_idx]
            X_batch, t_batch = X_train[batch_indices], t_train[batch_indices]

            # Update weights
            dw = grad_multiclass(w, X_batch, t_batch)
            w -= alpha * dw

            # Optionally, calculate loss and accuracy for plotting
            if plot:
                train_loss.append(loss_multiclass(w, X_train, t_train))
                valid_loss.append(loss_multiclass(w, X_valid, t_valid))
                train_acc.append(accuracy_multiclass(w, X_train, np.argmax(t_train, axis=1)))
                valid_acc.append(accuracy_multiclass(w, X_valid, np.argmax(t_valid, axis=1)))

    # Plotting the training curves
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(valid_loss, label='Validation Loss')
        plt.title("Training and Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(valid_acc, label='Validation Accuracy')
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # Return the final weights
    return w
