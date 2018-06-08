import numpy as np
import matplotlib.pyplot as plt
import pickle
from mnist import MNIST
from sklearn.datasets import fetch_mldata
from numpy import random
from numpy import arange
#sigmoid activatiob function
from GeneticAlgorithm import *

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_deriv = lambda x: sigmoid(x) * (1-sigmoid(x))

#relu activation function
relu = lambda x: np.maximum(0, x)
relu_deriv = lambda x: np.maximum(np.sign(x), 0)

tanh = lambda x: np.tanh(x)
tanh_deriv = lambda x: 1 - np.power(np.tanh(x), 2)

#softmax
def softmax(x):
 e_x = np.exp(x - np.max(x))
 return e_x / e_x.sum(axis=0)

class NN:
    #hyper parameters
    def __init__(self, hlayer1_size = 0, hlayer2_size = 0, epochs = 0, lr = 0, active_func = None, active_func_deriv = None):
      self.hlayer1_size = hlayer1_size
      self.hlayer2_size = hlayer2_size
      self.W1 = np.random.uniform(-0.1, 0.1, (hlayer1_size, 784))
      self.b1 = np.random.uniform(-0.1, 0.1, hlayer1_size)
      self.W2 = np.random.uniform(-0.1, 0.1, (hlayer2_size, hlayer1_size))
      self.b2 = np.random.uniform(-0.1, 0.1, hlayer2_size)
      self.W3 = np.random.uniform(-0.1, 0.1, (10, hlayer2_size))
      self.b3 = np.random.uniform(-0.1, 0.1, 10)
      self.epochs = epochs
      self.lr = lr
      self.active_func = active_func
      self.active_func_deriv = active_func_deriv

    def clone(self, network):
        self.hlayer1_size = network.hlayer1_size
        self.hlayer2_size = network.hlayer2_size
        self.W1 = network.W1
        self.b1 = network.b1
        self.W2 = network.W2
        self.b2 = network.b2
        self.W3 = network.W3
        self.b3 = network.b3
        self.epochs = network.epochs
        self.lr = network.lr
        self.active_func = network.active_func
        self.active_func_deriv = network.active_func_deriv

    #foward propagation
    def fprop(self, x):
      z1 = np.dot(self.W1, x) + self.b1
      h1 = self.active_func(z1)
      z2 = np.dot(self.W2, h1) + self.b2
      h2 = self.active_func(z2)
      z3 = np.dot(self.W3, h2) + self.b3
      h3 = softmax(z3)
      ret = {'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3':z3, 'h3':h3}
      return ret

    def get_test_acc(self, test_x, test_y):
        indexes = list(range(test_x.shape[0]))
        correct = 0.0
        for i in indexes:
          fprop_cache = self.fprop(test_x[i])
          prediction = str(np.argmax(fprop_cache['h3']))
          if(test_y[i] == int(prediction)):
            correct = correct + 1
        accuracy = 100.0*correct/len(test_x)
        # print("test accuracy = " + str(accuracy))
        return accuracy

    #backward propagation
    def bprop(self, x, y, fprop_cache):
      z1, h1, z2, h2, z3, h3 = [fprop_cache[key] for key in ('z1', 'h1', 'z2', 'h2', 'z3', 'h3')]

      db3 = np.copy(h3)
      db3[int(y)] -= 1
      dW3 = np.outer(db3, h2.T)
      dh2 = db3.T.dot(self.W3)
      db2 = dh2.T * active_func_deriv(z2)
      dW2 = db2.reshape(len(db2),1).dot(h1.reshape(len(h1),1).T)
      dh1 = db2.T.dot(self.W2)
      db1 = dh1.T * active_func_deriv(z1)
      dW1 = db1.reshape(len(db1),1).dot(x.reshape(len(x), 1).T)

      '''tmp = np.copy(h3).dot(self.W3) - self.W3[int(y), :]
      derivative = self.active_func_deriv(z2)

      db2 = tmp * derivative
      dW2 = np.outer(db2, h1)

      tmp2 = np.copy(h2).dot(self.W2) - self.W2[int(y), :]
      derivative2 = self.active_func_deriv(z1)

      db1 = tmp2 * derivative2
      dW1 = np.outer(db1, x)'''

      return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3':dW3 , 'b3':db3}


    #update weights schotastic gradient descent
    def update_weights_sgd(self, bprop_cache):
      dW1, db1, dW2, db2, dW3, db3 = [bprop_cache[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
      self.W1 -= dW1 * self.lr
      self.b1 -= db1 * self.lr
      self.W2 -= dW2 * self.lr
      self.b2 -= db2 * self.lr
      self.W3 -= dW3 * self.lr
      self.b3 -= db3 * self.lr

    #This is the function where the training is done
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

  # Define training and testing sets
  #indices = arange(len(mnist.data))
  #random.seed(0)

  train_idx = arange(0, n_train)
  test_idx = arange(n_train + 1, n_train + n_test)

  train_x, train_y = mnist.data[train_idx], mnist.target[train_idx]
  test_x, test_y = mnist.data[test_idx], mnist.target[test_idx]
  train_x = train_x / 255.0
  test_x = test_x / 255.0

  # seed = np.random.randint(1, 101)
  # np.random.seed(seed)
  np.random.shuffle(train_x)
  # np.random.seed(seed)
  np.random.shuffle(train_y)

  valid_size = int(train_x.shape[0] * 0.2)
  valid_x, valid_y = train_x[:valid_size], train_y[:valid_size]
  train_x, train_y = train_x[valid_size:], train_y[valid_size:]



  return train_x, train_y, valid_x, valid_y, test_x, test_y

if __name__ == '__main__':
  '''train_x = np.loadtxt("train_x")
      train_y = np.loadtxt("train_y")
      test_x = np.loadtxt("test_x")

      with open("pickle_train_x", 'wb') as f1:
        pickle.dump(train_x, f1)
      with open("pickle_train_y", 'wb') as f2:
        pickle.dump(train_y, f2)
      with open("pickle_test_x", 'wb') as f3:
        pickle.dump(test_x, f3)'''

  '''with open("pickle_train_x", 'rb') as f1:
    train_x = pickle.load(f1)
  with open("pickle_train_y", 'rb') as f2:
    train_y = pickle.load(f2)
  with open("pickle_test_x", 'rb') as f3:
    test_x = pickle.load(f3)
  #normalization'''

  train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()

  # split the training data into 80% training and 20% validation

  active_func = tanh
  active_func_deriv = tanh_deriv
  epochs =  50
  lr = 0.01
  hlayer1_size = 128
  hlayer2_size = 64



  nn = NN(hlayer1_size, hlayer2_size, epochs, lr, active_func, active_func_deriv)

  weights_amount = 784 * hlayer1_size + hlayer1_size + hlayer1_size * hlayer2_size + hlayer2_size + hlayer2_size * 10 + 10

  # rnd_test_inds = random.sample(range(len(test_y)),100)
  # g_test_x = [test_x[i] for i in rnd_test_inds]
  # g_test_y = [test_y[i] for i in rnd_test_inds]
  # g_test = [g_test_x, g_test_y]

  g_test = [test_x[:1000],test_y[:1000]]

  g = Genetics(weights_amount, retain=0.7, random_select=0.001, mutate_chance=0.08, network=nn, test = g_test)
  g.create_population(50)
  g.breed(g.population[0], g.population[1])
  g.run(100)
  # nn.train(train_x, train_y, valid_x, valid_y)
  # nn.get_test_acc(test_x,test_y)
