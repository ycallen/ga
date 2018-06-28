import numpy as np
import random
import sys

import time

from nn import *
from utils import *
import pickle

class Chromosome:
    def __init__(self, layers):
        self.W1 = self.initialize_weight(784,layers[0])
        self.b1 = self.initialize_weight(layers[0])
        self.W2 = self.initialize_weight(layers[0],layers[1])
        self.b2 = self.initialize_weight(layers[1])
        self.W3 = self.initialize_weight(layers[1],10)
        self.b3 = self.initialize_weight(10)

    def initialize_weight(self, s1, s2=None):
        eps = np.sqrt(6.0 / (s1 + (s2 if s2 is not None else 1)))
        if s2 is None:
            return np.random.uniform(-eps, eps, s1)
        return np.random.uniform(-eps, eps, (s1, s2))

    def mutate(self, mutate_rate, size):
        """
        Randomly mutate one part of the network.
        """
        self.W1 = self.W1 + np.random.binomial(1, mutate_rate) * np.random.normal(0, size, size=self.W1.shape)
        self.W2 = self.W2 + np.random.binomial(1, mutate_rate) * np.random.normal(0, size, size=self.W2.shape)
        self.W3 = self.W3 + np.random.binomial(1, mutate_rate) * np.random.normal(0, size, size=self.W3.shape)
        self.b1 = self.b1 + np.random.binomial(1, mutate_rate) * np.random.normal(0, size, size=self.b1.shape[0])
        self.b2 = self.b2 + np.random.binomial(1, mutate_rate) * np.random.normal(0, size, size=self.b2.shape[0])
        self.b3 = self.b3 + np.random.binomial(1, mutate_rate) * np.random.normal(0, size, size=self.b3.shape[0])

    def calc_fitness(self, network, train, size=100):

        network.W1 = self.W1
        network.W2 = self.W2
        network.W3 = self.W3
        network.b1 = self.b1
        network.b2 = self.b2
        network.b3 = self.b3

        #rnd_indices = random.sample(range(len(train[1])), size)
        #train_x = np.asarray([train[0][i] for i in rnd_indices])
        #train_y = np.asarray([train[1][i] for i in rnd_indices])
        train_x, train_y = train

        val = network.get_acc_and_loss(train_x, train_y)
        return val


def select_parents(graded, selection_type):
    #ranking selection
    weights = range(len(graded), 0, -1)
    sum_weights = sum(weights)
    weights = [float(w)/float(sum_weights) for w in weights]
    val  = np.random.choice([element[1] for element in graded], 2, True, weights)
    return val


class Genetics:
    def __init__(self, hidden_layers_sz, retain, random_select, mutate_chance, network, train, validation, test,
                 activation, mutate_size):
        self.population = []
        self.hidden_layers_sz = hidden_layers_sz
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.inner_network = NN(hidden_layers_sz=hidden_layers_sz)
        self.inner_network.clone(network)
        self.test = test
        self.train = train
        self.validation = validation
        self.best_chrom = (-1, None)
        self.activation = activation
        self.best_devel = (-1, None)
        self.mutate_size = mutate_size

        print "retain = " + str(retain)
        print "mutate_chance = " + str(mutate_chance)
        print "mutate_size = " + str(mutate_size)
        print "random select = " + str(random_select)

    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        """
        pop = []
        for _ in range(0, count):
            # Create a random chromosome, which is all the weights of the network
            # Add the network to our population.
            pop.append(Chromosome(self.hidden_layers_sz))

        self.population = pop
        return pop

    def crossover_param(self,child_param,p1_param,p2_param):
        #crossover of either weight or bias
        if p1_param.shape[0] == 1:
            if np.random.random() < 0.5:
                child_param = p1_param
            else:
                child_param = p2_param
        else:
            for i in xrange(p1_param.shape[0]):
                if np.random.random() < 0.5:
                    child_param[i] =  p1_param[i]
                else:
                    child_param[i] = p2_param[i]

    def crossover(self,p1, p2):
        #crossover by row
        child = Chromosome(self.hidden_layers_sz)
        self.crossover_param(child.W1, p1.W1, p2.W1)
        self.crossover_param(child.b1, p1.b1, p2.b1)
        self.crossover_param(child.W2, p1.W2, p2.W2)
        self.crossover_param(child.b2, p1.b2, p2.b2)
        self.crossover_param(child.W3, p1.W3, p2.W3)
        self.crossover_param(child.b3, p1.b3, p2.b3)
        return child

    def evolve(self):
        """Evolve a population of chromosomes.
        Args:
            pop (list): A list of network parameters
        """

        # get activation and derivative functions
        self.inner_network.active_func, self.inner_network.active_func_deriv = self.activation

        rnd_indices = random.sample(range(len(self.train[1])), 100)
        train_x = np.asarray([self.train[0][i] for i in rnd_indices])
        train_y = np.asarray([self.train[1][i] for i in rnd_indices])
        train = (train_x, train_y)
        # Get scores for each network7
        ranked = [(chrom.calc_fitness(self.inner_network, train, 100), chrom) for chrom in self.population]
        graded = [(r[0][0], r[1]) for r in list(ranked)]

        # Sort on the scores.
        graded = [x for x in sorted(graded, key=lambda g: g[0], reverse=True)]

        graded_copy = list(graded)

        print "avg acc: {:^3.2f} avg loss: {:^3.2f} max acc: {:^3.2f}\n".format(np.mean([r[0][0] for r in ranked]),
                                                     np.mean([r[0][1] for r in ranked]),
                                                     graded[0][0]),

        graded_only_chrom = [x[1] for x in list(graded)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded_copy) * self.retain)

        # The parents are every network we want to keep.
        new_pool = graded_only_chrom[:retain_length]

        # save the best in each iteration
        self.best_chrom = graded[0]

        for individual in graded_only_chrom[retain_length:]:
            if self.random_select > random.random():
                new_pool.append(individual)

        # Now findcalc out how many spots we have left to fill.
        desired_length = len(self.population) - len(new_pool)

        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            parents = select_parents(graded_copy, 'ranking')

            p1 = parents[0]
            p2 = parents[1]

            # create child, and with probability mutate it
            child = self.crossover(p1, p2)
            child.mutate(self.mutate_chance, self.mutate_size)

            # Add the children one at a time.
            if len(children) < desired_length:
                children.append(child)

        new_pool.extend(children)
        self.population = list(new_pool)

    def run(self, iterations):
        for i in xrange(iterations):
            print str(i)+":",
            self.evolve()

            if i % 100 == 0 and i > 0:
                self.validate_on_test()

        # print best to file
        name = "best_devel_weights_" + str(time.time())
        print "writing to file " + name
        with open(name, 'wb') as file:
            pickle.dump(self.best_devel[1], file)
        # with open("best_devel_weights_1529639442.52", 'r') as file:
        #     temp = pickle.load(file)
        #     print "best_on_test: {:^3.2f}".format(
        #         temp.calc_fitness(self.inner_network, self.test, len(self.test[0]))[0])


    def validate_on_test(self):
        print "*******************************************************************"
        print "DEVEL :",
        latest_devel_acc = self.best_chrom[1].calc_fitness(self.inner_network, self.validation, len(self.validation[0]))[0]
        print "best_on_devel: {:^3.2f}".format(latest_devel_acc)
        if latest_devel_acc >= self.best_devel[0]:
            self.best_devel = (latest_devel_acc, self.best_chrom[1])
        print "TEST  :",
        print "best_on_test: {:^3.2f}".format(self.best_chrom[1].calc_fitness(self.inner_network, self.test, len(self.test[0]))[0])
        print "max acc: {:^3.2f}\n".format(self.best_chrom[0]),
        print "*******************************************************************"


def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()
    # split the training data into 80% training and 20% validation
    activation = [tanh, tanh_deriv]
    hidden_layers_sz = [128, 64]

    nn = NN(hidden_layers_sz=hidden_layers_sz, activation=activation)

    train_set = [train_x, train_y]
    valid_set = [valid_x, valid_y]
    test_set = [test_x, test_y]

    # if command line arguments are given, use them
    if len(sys.argv) == 6:
        print "parsing command line arguments..."
        population_size = int(sys.argv[1])
        retain = float(sys.argv[2])
        random_select = float(sys.argv[3])
        mutate_change = float(sys.argv[4])
        mutate_size = float(sys.argv[5])
    else:
        retain = 0.2
        random_select = 0.03
        mutate_change = 0.2
        mutate_size = 0.012
        population_size = 150

    g = Genetics(hidden_layers_sz=hidden_layers_sz,
                 retain=retain,
                 random_select=random_select,
                 mutate_chance=mutate_change,
                 mutate_size=mutate_size,
                 network=nn,
                 train=train_set,
                 validation=valid_set,
                 test=test_set,
                 activation=(tanh, tanh_deriv))


    print "pop size = " + str(population_size)
    g.create_population(population_size)
    g.run(10000)

    # nn.train(train_x, train_y, valid_x, valid_y)
    # nn.get_test_acc(test_x,test_y)


if __name__ == '__main__':
    main()
