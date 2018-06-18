from sklearn.datasets import fetch_mldata
from numpy import arange
import random
import numpy as np
import collections


# sigmoid activation function
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_deriv = lambda x: sigmoid(x) * (1 - sigmoid(x))

# relu activation function
relu = lambda x: np.maximum(0, x)
relu_deriv = lambda x: np.maximum(np.sign(x), 0)

# tanh activation function
tanh = lambda x: np.tanh(x)
tanh_deriv = lambda x: 1 - np.power(np.tanh(x), 2)


# softmax

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_data():
    mnist = fetch_mldata('MNIST original')

    n_train = 60000
    n_test = 10000

    train_idx = arange(0, n_train)
    random.shuffle(train_idx)
    test_idx = arange(n_train, n_train + n_test)
    random.shuffle(test_idx)

    train_x, train_y = mnist.data[train_idx], mnist.target[train_idx]
    test_x, test_y = mnist.data[test_idx], mnist.target[test_idx]
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    valid_size = int(train_x.shape[0] * 0.2)
    valid_x, valid_y = train_x[:valid_size], train_y[:valid_size]
    train_x, train_y = train_x[valid_size:], train_y[valid_size:]

    return train_x, train_y, valid_x, valid_y, test_x, test_y