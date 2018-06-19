import numpy as np
import random
from nn import *
from utils import *

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
        self.best_chrom = (-1, None)
        self.activation = activation
        self.by_loss = by_loss

        print "retain = " + str(retain)
        print "mutate_chance = " + str(mutate_chance)
        print "activation_options = " + str(activation)

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

        # Get scores for each network7
        ranked = [(chrom.calc_fitness(self.inner_network, self.train, 150), chrom) for chrom in self.population]
        graded = [(r[0][0], r[1]) for r in list(ranked)]

        # Sort on the scores.
        graded = [x for x in sorted(graded, key=lambda g: g[0], reverse=True)]

        graded_copy = list(graded)

        # update best entity and update graded
        if graded[0][0] > self.best_chrom[0]:
            self.best_chrom = graded[0]

        print "avg acc: {:^3.2f} avg loss: {:^3.2f} max acc: {:^3.2f}\n".format(np.mean([r[0][0] for r in ranked]),
                                                     np.mean([r[0][1] for r in ranked]),
                                                     self.best_chrom[0]),

        graded_only_chrom = [x[1] for x in list(graded)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded_copy) * self.retain)

        # The parents are every network we want to keep.
        new_pool = graded_only_chrom[:retain_length]

        # Now find out how many spots we have left to fill.
        desired_length = len(self.population) - len(new_pool)

        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            parents = select_parents(graded_copy, 'ranking')

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

            if (i % 10  == 0 and i > 0):
                self.validate_on_test()


    def validate_on_test(self):
        print "*******************************************************************"
        print "DEVEL :",
        print "best_on_devel: {:^3.2f}".format(self.best_chrom[1].calc_fitness(self.inner_network, self.validation, len(self.validation[0]))[0])
        print "TEST  :",
        print "best_on_test: {:^3.2f}".format(self.best_chrom[1].calc_fitness(self.inner_network, self.test, len(self.test[0]))[0])
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


    g = Genetics(hidden_layers_sz = hidden_layers_sz, retain=0.06, random_select=0.00, mutate_chance=0.02, network=nn, train=train_set,
                 validation=valid_set, test=test_set, activation=(tanh, tanh_deriv), by_loss=False)

    population_size = 100
    print "pop size = " + str(population_size)
    g.create_population(population_size)
    g.run(10000)

    # nn.train(train_x, train_y, valid_x, valid_y)
    # nn.get_test_acc(test_x,test_y)
if __name__ == '__main__':
    main()