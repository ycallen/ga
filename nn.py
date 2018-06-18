from sklearn.datasets import fetch_mldata
from numpy import arange
import numpy as np
from utils import *

class NN:
    # hyper parameters
    def __init__(self, hidden_layers_sz, epochs=0, lr=0, activation=None):
        self.hidden_layers_sz = hidden_layers_sz
        self.W1 = np.random.uniform(-0.1, 0.1, (hidden_layers_sz[0], 784))
        self.b1 = np.random.uniform(-0.1, 0.1, hidden_layers_sz[0])
        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_layers_sz[1], hidden_layers_sz[0]))
        self.b2 = np.random.uniform(-0.1, 0.1, hidden_layers_sz[1])
        self.W3 = np.random.uniform(-0.1, 0.1, (10, hidden_layers_sz[1]))
        self.b3 = np.random.uniform(-0.1, 0.1, 10)
        self.epochs = epochs
        self.activation = activation
        self.lr = lr

    def clone(self, network):
        self.hidden_layers_sz = network.hidden_layers_sz
        self.W1 = network.W1
        self.b1 = network.b1
        self.W2 = network.W2
        self.b2 = network.b2
        self.W3 = network.W3
        self.b3 = network.b3
        self.epochs = network.epochs
        self.lr = network.lr
        self.activation = network.activation

    # foward propagation
    def fprop(self, x):
        active_func = self.activation[0]
        z1 = np.dot(x, self.W1) + self.b1
        h1 = active_func(z1)
        z2 = np.dot(h1, self.W2) + self.b2
        h2 = active_func(z2)
        z3 = np.dot(h2, self.W3) + self.b3
        h3 = softmax(z3)
        ret = {'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3, 'h3': h3}
        return ret

    def get_acc_and_loss(self, test_x, test_y):

        indexes = list(range(test_x.shape[0]))
        correct = 0.0
        total_loss = 0.0
        for i in indexes:
            fprop_cache = self.fprop(test_x[i])
            prediction = str(np.argmax(fprop_cache['h3']))
            y = test_y[i]
            if y == int(prediction):
                correct = correct + 1
            loss = -np.log(fprop_cache['h3'][int(y)])
            total_loss += loss

        accuracy = 100.0 * correct / len(test_x)
        loss = total_loss / len(test_x)

        return accuracy, loss


    # backward propagation
    def bprop(self, x, y, fprop_cache):
        z1, h1, z2, h2, z3, h3 = [fprop_cache[key] for key in ('z1', 'h1', 'z2', 'h2', 'z3', 'h3')]
        active_deriv_func = self.activation[1]
        db3 = np.copy(h3)
        db3[int(y)] -= 1
        dW3 = np.outer(db3, h2.T)
        dh2 = db3.T.dot(self.W3)
        db2 = dh2.T * active_deriv_func(z2)
        dW2 = db2.reshape(len(db2), 1).dot(h1.reshape(len(h1), 1).T)
        dh1 = db2.T.dot(self.W2)
        db1 = dh1.T * active_deriv_func(z1)
        dW1 = db1.reshape(len(db1), 1).dot(x.reshape(len(x), 1).T)


        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    # update weights schotastic gradient descent
    def update_weights_sgd(self, bprop_cache):
        dW1, db1, dW2, db2, dW3, db3 = [bprop_cache[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
        self.W1 -= dW1 * self.lr
        self.b1 -= db1 * self.lr
        self.W2 -= dW2 * self.lr
        self.b2 -= db2 * self.lr
        self.W3 -= dW3 * self.lr
        self.b3 -= db3 * self.lr

    def train(self, train_x, train_y, valid_x, valid_y):
        for epoch in range(self.epochs):  # for each epoch:

            train_sum_loss = 0.0
            train_correct = 0

            valid_correct = 0

            indexes = list(range(train_x.shape[0]))
            np.random.shuffle(indexes)

            for i in indexes:
                x, y = train_x[i], train_y[i]
                fprop_cache = self.fprop(x)
                loss = -np.log(fprop_cache['h3'][int(y)])
                train_sum_loss += loss
                if np.argmax(fprop_cache['h3']) == int(y):
                    train_correct += 1
                bprop_cache = self.bprop(x, y, fprop_cache)
                params = self.update_weights_sgd(bprop_cache)  # updates the weights

            for x, y in zip(valid_x, valid_y):
                fprop_cache = self.fprop(x)
                if np.argmax(fprop_cache['h3']) == int(y):
                    valid_correct += 1

            print("Epoch # %d : training accuracy = %f, valid accuracy = %f, train_loss = %f" % \
                  (epoch,
                   float(train_correct) / len(train_x),
                   float(valid_correct) / len(valid_x),
                   train_sum_loss))


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

def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()
    # split the training data into 80% training and 20% validation
    activation = [tanh, tanh_deriv]
    epochs = 50
    lr = 0.01
    hidden_layers_sz = [128, 64]

    nn = NN(hidden_layers_sz=hidden_layers_sz, activation=activation)

    train_set = [train_x, train_y]
    valid_set = [valid_x, valid_y]
    test_set = [test_x, test_y]
    layers = [784, 128, 64, 10]

    #g = Genetics(layers=layers, retain=0.05, random_select=0.00, mutate_chance=0.01, network=nn, train=train_set,
    #             validation=valid_set, test=test_set, activation=(tanh, tanh_deriv), by_loss=False)
    population_size = 100
    print "pop size = " + str(population_size)
    #g.create_population(population_size)
    #g.run(10000)

    # nn.train(train_x, train_y, valid_x, valid_y)
    # nn.get_test_acc(test_x,test_y)
if __name__ == '__main__':
    main()
