import numpy as np
import random
import sys
from nn import *
from utils import *

class Chromosome:
    def __init__(self, hidden_layers_sz = [128, 64], accuracy = sys.float_info.min, loss = sys.float_info.max):
        self.W1 = self.initialize_weight(784,hidden_layers_sz[0])
        self.b1 = self.initialize_weight(hidden_layers_sz[0])
        self.W2 = self.initialize_weight(hidden_layers_sz[0],hidden_layers_sz[1])
        self.b2 = self.initialize_weight(hidden_layers_sz[1])
        self.W3 = self.initialize_weight(hidden_layers_sz[1],10)
        self.b3 = self.initialize_weight(10)
        self.accuracy = accuracy
        self.loss = loss

    def initialize_weight(self, s1, s2=None):
        eps = np.sqrt(6.0 / (s1 + (s2 if s2 is not None else 1)))
        if s2 is None:
            return np.random.uniform(-eps, eps, s1)
        return np.random.uniform(-eps, eps, (s1, s2))

    def mutate(self, mutate_rate):
        """
        Randomly mutate one part of the network.
        """
        self.W1 = self.W1 + np.random.binomial(1, mutate_rate) * np.random.normal(0, 0.1, size=self.W1.shape)
        self.W2 = self.W2 + np.random.binomial(1, mutate_rate) * np.random.normal(0, 0.1, size=self.W2.shape)
        self.W3 = self.W3 + np.random.binomial(1, mutate_rate) * np.random.normal(0, 0.1, size=self.W3.shape)
        self.b1 = self.b1 + np.random.binomial(1, mutate_rate) * np.random.normal(0, 0.1, size=self.b1.shape[0])
        self.b2 = self.b2 + np.random.binomial(1, mutate_rate) * np.random.normal(0, 0.1, size=self.b2.shape[0])
        self.b3 = self.b3 + np.random.binomial(1, mutate_rate) * np.random.normal(0, 0.1, size=self.b3.shape[0])

    def calc_fitness(self, network, test, size=100):

        network.W1 = self.W1
        network.W2 = self.W2
        network.W3 = self.W3
        network.b1 = self.b1
        network.b2 = self.b2
        network.b3 = self.b3

        rnd_indices = random.sample(range(len(test[1])), size)
        train_x = np.asarray([test[0][i] for i in rnd_indices])
        train_y = np.asarray([test[1][i] for i in rnd_indices])

        metrics = network.get_acc_and_loss(train_x, train_y)
        self.accuracy = metrics[0]
        self.loss = metrics[1]
        return self


def select_parents(chromosones):
    #ranking selection
    weights = range(len(chromosones), 0, -1)
    sum_weights = sum(weights)
    weights = [float(w)/float(sum_weights) for w in weights]
    val  = np.random.choice([c for c in chromosones], 2, True, weights)
    return val


class Genetics:
    def __init__(self, hidden_layers_sz, retain, random_select, mutate_chance, network, train, validation, test,
                 activation, by_loss):
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
        self.best_chrom = Chromosome()
        self.activation = activation
        self.by_loss = by_loss

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

    def crossover_param(self,child_param,father_param,mother_param):
        #crossover of weight or bias
        if father_param.shape[0] == 1:
            if np.random.random() < 0.5:
                child_param = father_param
            else:
                child_param = mother_param
        else:
            for i in xrange(father_param.shape[0]):
                if np.random.random() < 0.5:
                    child_param[i] =  father_param[i]
                else:
                    child_param[i] = mother_param[i]

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

        #Get all metrics for each network
        chromosones = [chrom.calc_fitness(self.inner_network, self.train) for chrom in self.population]


        #Sort chromosones by accuracy
        chromosones = [x for x in sorted(chromosones, key=lambda g: g.accuracy, reverse=True)]

        # update best chromosone
        if chromosones[0].accuracy > self.best_chrom.accuracy:
            self.best_chrom = Chromosome(accuracy=chromosones[0].accuracy, loss=chromosones[0].loss)

        print "avg acc: {:^3.2f} highest acc: {:^3.2f}\n".format(np.mean([c.accuracy for c in chromosones]),
                                                                 self.best_chrom.accuracy),


        chromosones = list(chromosones)

        accuracies = [c.accuracy for c in chromosones]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(accuracies) * self.retain)

        # The parents are every network we want to keep.
        new_pool = chromosones[:retain_length]

        # Now find out how many spots we have left to fill.
        desired_length = len(self.population) - len(new_pool)

        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            parents = select_parents(chromosones)

            p1 = parents[0]
            p2 = parents[1]

            child = self.crossover(p1, p2)

            # Add the children one at a time.
            if len(children) < desired_length:
                children.append(child)

        new_pool.extend(children)
        for individual in new_pool:
            individual.mutate(self.mutate_chance)

        self.population = list(new_pool)

    def run(self, iterations):
        for i in xrange(iterations):
            print str(i)+":",
            self.evolve()

            if (i % 100  == 0 and i > 0):
                self.validate_on_test()


    def validate_on_test(self):
        print "*******************************************************************"
        print "DEVEL :",
        print "best_on_devel : {:^3.2f}".format(self.best_chrom.calc_fitness(self.inner_network, self.validation, len(self.validation[0])).accuracy)
        print "TEST  :",
        print "best_on_test  : {:^3.2f}".format(self.best_chrom.calc_fitness(self.inner_network, self.test, len(self.test[0])).accuracy)
        print "*******************************************************************"
        # ranked = [chrom.fitness(self.inner_network, self.test, size=1000) for chrom in self.population]
        #
        # print " : avg: {: ^3.2f}, max: {:^3.2f}, best_on_test: {:^3.2f}".format(np.mean(ranked),
        #                         np.max(ranked), self.best_chrom[1].fitness(self.inner_network, self.test, size=1000)[0])

def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()
    # split the training data into 80% training and 20% validation
    activation = [tanh, tanh_deriv]
    hidden_layers_sz = [128, 64]

    nn = NN(hidden_layers_sz= hidden_layers_sz, activation= activation)

    train_set = [train_x, train_y]
    valid_set = [valid_x, valid_y]
    test_set = [test_x, test_y]


    g = Genetics(hidden_layers_sz = hidden_layers_sz, retain=0.05, random_select=0.00, mutate_chance=0.01, network=nn, train=train_set,
                 validation=valid_set, test=test_set, activation=(tanh, tanh_deriv), by_loss=False)

    population_size = 100
    print "pop size = " + str(population_size)
    g.create_population(population_size)
    g.run(10000)

    # nn.train(train_x, train_y, valid_x, valid_y)
    # nn.get_test_acc(test_x,test_y)
if __name__ == '__main__':
    main()