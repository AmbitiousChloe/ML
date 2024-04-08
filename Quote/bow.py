import numpy as np


def insert_feature(nparray, vocab):
    wl = nparray[:, 3]
    x = np.zero((nparray.shape[0], len(vocab)))
    for i in range(nparray.shape[0]):
        for j in range(len(vocab)):
            if vocab[j] in wl[i]:
                x[i, j] = 1
    return x


















