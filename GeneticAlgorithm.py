import numpy as np
import random
from ex1 import *


class Chromosome:
    def __init__(self, length=0):
        self.W1 = self.initialize_weight(784,128)
        self.b1 = self.initialize_weight(128)
        self.W2 = self.initialize_weight(128,64)
        self.b2 = self.initialize_weight(64)
        self.W3 = self.initialize_weight(64,10)
        self.b3 = self.initialize_weight(10)

    def initialize_weight(self, s1, s2=None):
        eps = np.sqrt(6.0 / (s1 + (s2 if s2 is not None else 1)))
        if s2 is None:
            return np.random.uniform(-eps, eps, s1)
        return np.random.uniform(-eps, eps, (s1, s2))

    def mutate(self):
        """
        Randomly mutate one part of the network.
        """
        self.W1 = self.W1 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.1, size=self.W1.shape)
        self.W2 = self.W2 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.1, size=self.W2.shape)
        self.W3 = self.W3 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.1, size=self.W3.shape)
        self.b1 = self.b1 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.1, size=self.b1.shape[0])
        self.b2 = self.b2 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.1, size=self.b2.shape[0])
        self.b3 = self.b3 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.1, size=self.b3.shape[0])

    def fitness(self, network, test):

        network.W1 = self.W1
        network.W2 = self.W2
        network.W3 = self.W3
        network.b1 = self.b1
        network.b2 = self.b2
        network.b3 = self.b3

        rnd_indices = random.sample(range(len(test[1])), 100)
        train_x = np.asarray([test[0][i] for i in rnd_indices])
        train_y = np.asarray([test[1][i] for i in rnd_indices])

        val = network.get_acc_and_loss(train_x, train_y)
        return val


def select_parents(graded, selection_type):
    weights = []
    if selection_type == 'roulette':
        weights = [element[0] for element in graded]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]  # normalize weights
    elif selection_type == 'ranking':
        weights = range(len(graded), 0, -1)
        sum_weights = sum(weights)
        weights = [float(w)/float(sum_weights) for w in weights]
    val  = np.random.choice([element[1] for element in graded], 2, True, weights)
    return val


class Genetics:
    def __init__(self, weights_sizes, retain, random_select, mutate_chance, network, train, test, activation_options,
                 by_loss):
        self.chromosome_length = weights_sizes
        self.population = []
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.inner_network = NN()
        self.inner_network.clone(network)
        self.test = test
        self.train = train
        self.best_chrom = (-1, None)
        self.activation_options = activation_options
        self.by_loss = by_loss

        print "retain = " + str(retain)
        print "mutate_chance = " + str(mutate_chance)
        print "activation_options = " + str(activation_options)

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
            pop.append(Chromosome(self.chromosome_length))

        self.population = pop
        return pop

    def crossover_param(self,child_param,father_param,mother_param):
        for i in xrange(father_param.shape[0]):
            if np.random.random() < 0.5:
                child_param[i] =  father_param[i]
            else:
                child_param[i] = mother_param[i]

    def crossover(self,father, mother, type="single_point"):
        """Make two children as parts of their parents.
        Args:
            mother (chromosome): Network weights
            father (chromosome): Network weights
        """
        children = []

        if type == "n_points_weights":
            c = Chromosome()
            self.crossover_param(c.W1, father.W1, mother.W1)
            self.crossover_param(c.b1, father.b1, mother.b1)
            self.crossover_param(c.W2, father.W2, mother.W2)
            self.crossover_param(c.b2, father.b2, mother.b2)
            self.crossover_param(c.W3, father.W3, mother.W3)
            self.crossover_param(c.b3, father.b3, mother.b3)
            children.append(c)
        return children

    def evolve(self):
        """Evolve a population of chromosomes.
        Args:
            pop (list): A list of network parameters
        """

        # select randomly 100 examples from the test collection. select the same 100 sample for each chromosome check
        # in generation


        # get random activation and derivative functions
        self.inner_network.active_func, self.inner_network.active_func_deriv = random.choice(self.activation_options)

        # Get scores for each network7
        ranked = [(chrom.fitness(self.inner_network, self.test), chrom) for chrom in self.population]
        graded = [(r[0][0], r[1]) for r in list(ranked)]

        print "acc|loss: {:^3.2f} | {:^3.2f}".format(np.mean([r[0][0] for r in ranked]), np.sum([r[0][1] for r in ranked])),

        # Sort on the scores.
        graded = [x for x in sorted(graded, key=lambda g: g[0], reverse=True)]
        print "max(acc): " + str(self.best_chrom[0])
        graded_copy = list(graded)

        # update best entity and update graded
        if graded[0][0] > self.best_chrom[0]:
            self.best_chrom = graded[0]
        graded_only_chrom = [x[1] for x in list(graded)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded_copy) * self.retain)

        # The parents are every network we want to keep.
        parents = graded_only_chrom[:retain_length]

        # # For those we aren't keeping, randomly keep some anyway.
        # for individual in graded_only_chrom[retain_length:]:
        #     if self.random_select > random.random():
        #         parents.append(individual)

        # Now find out how many spots we have left to fill.
        desired_length = len(self.population) - len(parents)

        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            # p_parents = select_parents(graded_copy[:-retain_length], random.choice(['roulette', 'ranking']))

            #p_parents = select_parents(graded_copy[:-retain_length], 'ranking')
            p_parents = select_parents(graded_copy, 'ranking')

            male = p_parents[0]
            female = p_parents[1]

            # Breed them.
            actions = ["n_points_weights"]
            c = random.choice(actions)
            babies = self.crossover(male, female, c)

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)

        parents.extend(children)
        for individual in parents:
            if self.mutate_chance > random.random():
                individual.mutate()

        self.population = list(parents)

    def run(self, iterations):
        for i in xrange(iterations):
            print str(i)+":",
            self.evolve()
