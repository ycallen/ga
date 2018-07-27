import pickle
import sys
import time
import random
import copy

from nn import *
from utils import *


def initialize_weight(s1, s2=None):
    eps = np.sqrt(6.0 / (s1 + (s2 if s2 is not None else 1)))
    if s2 is None:
        return np.random.uniform(-eps, eps, s1)
    return np.random.uniform(-eps, eps, (s1, s2))


class Chromosome:
    def __init__(self, layers):
        self.W1 = initialize_weight(784, layers[0])
        self.b1 = initialize_weight(layers[0])
        self.W2 = initialize_weight(layers[0], layers[1])
        self.b2 = initialize_weight(layers[1])
        self.W3 = initialize_weight(layers[1], 10)
        self.b3 = initialize_weight(10)

    def mutate(self, mutate_rate, size):
        """
        Randomly mutate one part of the network.
        """

        self.W1 += np.outer(np.random.binomial(1, mutate_rate, size=self.W1.shape[0]),
                            np.ones(self.W1.shape[1])) * np.random.normal(0, size, size=self.W1.shape)
        self.W2 += np.outer(np.random.binomial(1, mutate_rate, size=self.W2.shape[0]),
                            np.ones(self.W2.shape[1])) * np.random.normal(0, size, size=self.W2.shape)
        self.W3 += np.outer(np.random.binomial(1, mutate_rate, size=self.W3.shape[0]),
                            np.ones(self.W3.shape[1])) * np.random.normal(0, size, size=self.W3.shape)

        self.b1 += np.random.binomial(1, mutate_rate, size=self.b1.shape[0]) * np.random.normal(0, size,
                                                                                                size=self.b1.shape[0])
        self.b2 += np.random.binomial(1, mutate_rate, size=self.b2.shape[0]) * np.random.normal(0, size,
                                                                                                size=self.b2.shape[0])
        self.b3 += np.random.binomial(1, mutate_rate, size=self.b3.shape[0]) * np.random.normal(0, size,
                                                                                                size=self.b3.shape[0])

    def calc_fitness(self, network, train):
        network.W1 = self.W1
        network.W2 = self.W2
        network.W3 = self.W3
        network.b1 = self.b1
        network.b2 = self.b2
        network.b3 = self.b3

        # activations = [(tanh, tanh_deriv), (relu, relu_deriv), (sigmoid, sigmoid_deriv)]
        # network.activation = activations[np.random.choice(range(len(activations)))]
        train_x, train_y = train
        val = network.get_acc_and_loss(train_x, train_y)
        return val


def select_parents(graded, selection_type):
    # ranking selection
    weights = range(len(graded), 0, -1)
    sum_weights = sum(weights)
    weights = [float(w) / float(sum_weights) for w in weights]
    val = np.random.choice([element[1] for element in graded], 2, True, weights)
    return val


def batch_from(data, size):
    rnd_indices = random.sample(range(len(data[1])), size)
    train_x = np.asarray([data[0][i] for i in rnd_indices])
    train_y = np.asarray([data[1][i] for i in rnd_indices])
    return train_x, train_y


class Genetics:
    def __init__(self, hidden_layers_sz, retain, random_select, mutate_chance, network, train, validation, test,
                 activation, mutate_size, decay_factor):
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
        self.decay_factor = decay_factor

        print "retain = " + str(retain)
        print "mutate_chance = " + str(mutate_chance)
        print "mutate_size = " + str(mutate_size)
        print "random select = " + str(random_select)
        print "decay_factor = " + str(decay_factor)

    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        """
        self.population = [Chromosome(self.hidden_layers_sz) for _ in xrange(0, count)]

    @staticmethod
    def crossover_param(p1W, p2W, p1b, p2b):
        """ crossover by row weights and bias. Take the same row in W an in b, either from p1 or p2
        :param p1W: W of p1
        :param p2W: W of p2
        :param p1b: b of p1
        :param p2b: b of p2
        :return: the new weights matrix and new bias vector
        """

        v_b = np.random.randint(2, size=p1b.shape[0])
        new_b =  v_b * p1b + (1 - v_b) * p2b

        v_w = np.random.randint(2, size=p1W.shape[0])
        ones = np.ones(p1W.shape[1])
        m = v_w.reshape(v_w.shape[0], 1).dot(ones.reshape(1, ones.shape[0]))
        new_W =  p1W * m + p2W * (1 - m)

        return new_W, new_b

    def crossover(self, p1, p2):
        # crossover by row
        child = Chromosome(self.hidden_layers_sz)
        child.W1, child.b1 = self.crossover_param(p1.W1, p2.W1, p1.b1, p2.b1)
        child.W2, child.b2 = self.crossover_param(p1.W2, p2.W2, p1.b2, p2.b2)
        child.W3, child.b3 = self.crossover_param(p1.W3, p2.W3, p1.b3, p2.b3)
        return child

    def evolve(self):
        """
        Evolve a population of chromosomes.
        """

        # get activation and derivative functions
        self.inner_network.active_func, self.inner_network.active_func_deriv = self.activation

        # select train batch
        train = batch_from(data=self.train, size=100)

        # Get scores for each network, and sort
        ranked = [(chrom.calc_fitness(self.inner_network, train), chrom) for chrom in self.population]
        graded_acc = [(r[0][0], r[1]) for r in list(ranked)]
        graded_acc = [x for x in sorted(graded_acc, key=lambda g: g[0], reverse=True)]
        graded_loss = [(r[0][1], r[1]) for r in list(ranked)]
        graded_loss = [x for x in sorted(graded_loss, key=lambda g: g[0], reverse=False)]

        # print current state
        print "E[acc]: {:^3.2f} E[loss]: {:^3.2f} max[acc]: {:^3.2f} min[loss]: {:^3.2f}\n".format(np.mean([r[0][0] for r in ranked]),
                                                                                np.mean([r[0][1] for r in ranked]),
                                                                                graded_acc[0][0], graded_loss[0][0]),

        ranked_list = graded_loss # add condition!

        graded_only_chrom = [x[1] for x in list(ranked_list)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(ranked_list) * self.retain)

        # The parents are every network we want to keep.
        new_pool = copy.deepcopy(graded_only_chrom[:retain_length])

        # save the best in each iteration
        self.best_chrom = ranked_list[0]

        for individual in graded_only_chrom[retain_length:]:
            if self.random_select > random.random():
                new_pool.append(individual)

        # Now calc out how many spots we have left to fill.
        desired_length = len(self.population) - len(new_pool)

        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            parents = select_parents(ranked_list[:-retain_length], 'ranking')

            # create child, and with probability mutate it
            child = self.crossover(parents[0], parents[1])
            child.mutate(self.mutate_chance, self.mutate_size)

            # Add the children one at a time.
            if len(children) < desired_length:
                children.append(child)

        new_pool.extend(children)
        self.population = list(new_pool)

    def run(self, iterations):
        last_devel_acc = 0
        for i in xrange(iterations):
            print str(i) + ":",
            self.evolve()

            if i % 100 == 0 and i > 0:
                print i,
                self.validate_on_test()

            # decay mutate change
            if i % 1000 == 999:
                self.mutate_chance *= self.decay_factor

        # print best to output_weights_files
        name = "best_devel_weights_" + str(time.time())
        print "writing to output_weights_files " + name
        with open(name, 'wb') as output_weights_files:
            pickle.dump(self.best_devel[1], output_weights_files)

            # with open("best_devel_weights_1529639442.52", 'r') as output_weights_files:
            #     temp = pickle.load(output_weights_files)
            #     print "best_on_test: {:^3.2f}".format(
            #         temp.calc_fitness(self.inner_network, self.test, len(self.test[0]))[0])

    def validate_on_test(self):
        # print "*******************************************************************\n DEVEL :",
        latest_devel_acc = \
            self.best_chrom[1].calc_fitness(self.inner_network, self.validation)[0]
        if latest_devel_acc >= self.best_devel[0]: self.best_devel = (latest_devel_acc, self.best_chrom[1])
        print "+++ best_on_devel: {:^3.2f}, TEST  :best_on_test: {:^3.2f}, max acc: {:^3.2f} +++".format(
            latest_devel_acc, self.best_chrom[1].calc_fitness(self.inner_network, self.test)[0],self.best_chrom[0])
        return latest_devel_acc


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
    if len(sys.argv) == 7:
        print "parsing command line arguments..."
        population_size = int(sys.argv[1])
        retain = float(sys.argv[2])
        random_select = float(sys.argv[3])
        mutate_chance = float(sys.argv[4])
        mutate_size = float(sys.argv[5])
        decay_factor = float(sys.argv[6])
    else:
        retain = 0.1
        random_select = 0.0
        mutate_chance = 0.05
        mutate_size = 0.01
        population_size = 150
        decay_factor = 0.9

    g = Genetics(hidden_layers_sz=hidden_layers_sz,
                 retain=retain,
                 random_select=random_select,
                 mutate_chance=mutate_chance,
                 mutate_size=mutate_size,
                 network=nn,
                 train=train_set,
                 validation=valid_set,
                 test=test_set,
                 activation=(tanh, tanh_deriv),
                 decay_factor=decay_factor)

    print "pop size = " + str(population_size)
    g.create_population(population_size)
    g.run(20000)

    # nn.train(train_x, train_y, valid_x, valid_y)
    # nn.get_test_acc(test_x,test_y)


if __name__ == '__main__':
    main()
