import mnist
import numpy as np
from sklearn.utils import shuffle

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
    mnist_loader = mnist.MNIST(return_type="numpy")

    train_x, train_y = mnist_loader.load_training()
    train_x = train_x / 255.0

    test_x, test_y = mnist_loader.load_testing()
    test_x = test_x / 255.0

    train_x, train_y = shuffle(train_x, train_y)

    valid_size = 10000
    valid_x, valid_y = train_x[:valid_size], train_y[:valid_size]
    train_x, train_y = train_x[valid_size:], train_y[valid_size:]

    return train_x, train_y, valid_x, valid_y, test_x, test_y
