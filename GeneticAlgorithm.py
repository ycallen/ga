import numpy as np
import random
from ex1 import NN


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
        return np.random.uniform(-eps, eps, (s2, s1))

    def mutate(self):
        """
        Randomly mutate one part of the network.
        """
        self.W1 = self.W1 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.3, size=self.W1.shape)
        self.W2 = self.W2 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.3, size=self.W2.shape)
        self.W3 = self.W3 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.3, size=self.W3.shape)
        self.b1 = self.b1 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.3, size=self.b1.shape[0])
        self.b2 = self.b2 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.3, size=self.b2.shape[0])
        self.b3 = self.b3 + np.random.binomial(1, 0.1) * np.random.normal(0, 0.3, size=self.b3.shape[0])

    def fitness(self, network, test):

        network.W1 = self.W1
        network.W2 = self.W2
        network.W3 = self.W3
        network.b1 = self.b1
        network.b2 = self.b2
        network.b3 = self.b3

        val = network.get_acc_and_loss(test[0], test[1])
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
    return np.random.choice([element[1] for element in graded], 2, True, weights)


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

        print weights_sizes
        print retain
        print random_select
        print mutate_chance
        print activation_options

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

    def crossover(self, mother, father, type="single_point"):
        """Make two children as parts of their parents.
        Args:
            mother (chromosome): Network weights
            father (chromosome): Network weights
        """
        children = []

        if type == "n_points_weights":
            c = Chromosome()

            v = np.random.randint(2, size=father.W1.shape[0])
            ones = np.ones(father.W1.shape[1])
            m = v.reshape(v.shape[0], 1).dot(ones.reshape(1, ones.shape[0]))
            c.W1 = father.W1 * m + mother.W1 * (1 - m)
            c.b1 = v * father.b1 + (1 - v) * mother.b1

            v = np.random.randint(2, size=father.W2.shape[0])
            ones = np.ones(father.W2.shape[1])
            m = v.reshape(v.shape[0], 1).dot(ones.reshape(1, ones.shape[0]))
            c.W2 = father.W2 * m + mother.W2 * (1 - m)
            c.b2 = v * father.b2 + (1 - v) * mother.b2

            v = np.random.randint(2, size=father.W3.shape[0])
            ones = np.ones(father.W3.shape[1])
            m = v.reshape(v.shape[0], 1).dot(ones.reshape(1, ones.shape[0]))
            c.W3 = father.W3 * m + mother.W3 * (1 - m)
            c.b3 = v * father.b3 + (1 - v) * mother.b3

            children.append(c)
        return children

    def evolve(self):
        """Evolve a population of chromosomes.
        Args:
            pop (list): A list of network parameters
        """

        # select randomly 100 examples from the test collection. select the same 100 sample for each chromosome check
        # in generation
        # random.seed(1)
        rnd_indices = random.sample(range(len(self.train[1])), 100)
        train_x = np.asarray([self.train[0][i] for i in rnd_indices])
        train_y = np.asarray([self.train[1][i] for i in rnd_indices])

        # get random activation and derivative functions
        self.inner_network.active_func, self.inner_network.active_func_deriv = random.choice(self.activation_options)

        # Get scores for each network
        ranked = [(chrom.fitness(self.inner_network, [train_x, train_y]), chrom) for chrom in self.population]
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

            p_parents = select_parents(graded_copy[-retain_length:], 'ranking')

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
            random.seed(int(i/100.0))
            self.evolve()
