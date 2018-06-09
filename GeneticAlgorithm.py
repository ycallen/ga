import numpy as np
import random
from ex1 import NN

class Chromosome:
    def __init__(self, length):
        self.weights = np.random.uniform(0,1,length)

    def mutate(self):
        """
        Randomly mutate one part of the network.
        """
        self.weights += np.random.normal(0, 1, len(self.weights))

    def fitness(self, network, test):
        w1_values = self.weights[
                     0 :
                     network.hlayer1_size * 784]
        network.b1 = self.weights[
                     network.hlayer1_size * 784 :
                     network.hlayer1_size * 784 + network.hlayer1_size]
        w2_values = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size :
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size]
        network.b2 = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size :
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size + network.hlayer2_size ]
        w3_values = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size + network.hlayer2_size:
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size + network.hlayer2_size + network.hlayer2_size*10]
        network.b3 = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size + network.hlayer2_size + network.hlayer2_size * 10:
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size + network.hlayer2_size + network.hlayer2_size * 10 + 10]

        network.W1 = np.reshape(w1_values, (network.hlayer1_size, 784))
        network.W2 = np.reshape(w2_values, (network.hlayer2_size, network.hlayer1_size))
        network.W3 = np.reshape(w3_values, (10 ,network.hlayer2_size))

        #select randomly 100 examples from the test collection
        rnd_tests_indices = random.sample(range(len(test[1])),100)
        test_x = np.asarray([test[0][i] for i in rnd_tests_indices])
        test_y = np.asarray([test[1][i] for i in rnd_tests_indices])
        return network.get_test_acc(test_x, test_y)


class Genetics:
    def __init__(self, weights_sizes, retain, random_select, mutate_chance, network, test):
        self.chromosome_length = weights_sizes
        self.population = []
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.inner_network = NN()
        self.inner_network.clone(network)
        self.test = test
        self.best_chrom = (None,None)

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

    def breed(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (chromosome): Network weights
            father (chromosome): Network weights
        """
        children = []
        for _ in range(2):
            c = Chromosome(self.chromosome_length)
            for i in xrange(self.chromosome_length):
                c.weights[i] = random.choice([father.weights[i], mother.weights[i]])

            children.append(c)

        return children

    def fitness(self, network, test):
        return network.get_test_acc(test[0], test[1])

    def evolve(self):
        """Evolve a population of chromosomes.
        Args:
            pop (list): A list of network parameters
        """

        # Get scores for each network.
        graded = [(chrom.fitness(self.inner_network,self.test), chrom) for chrom in self.population]
        print "averaged fitness: " + str(np.mean([grade[0] for grade in graded])),
        print "best fitness: " + str(self.best_chrom[0])

        # Sort on the scores.
        graded = [x for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # update best entity
        if graded[0][0] > self.best_chrom[0]: self.best_chrom = graded[0]

        graded = [x[1] for x in graded]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Randomly mutate some of the networks we're keeping.
        for individual in parents:
            if self.mutate_chance > random.random():
                individual.mutate()

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(self.population) - parents_length
        assert desired_length >= 0
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same chromosome...
            #todo: can father and mother be the same?
            if male != female or True:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)
        self.population = parents

    def run(self, iterations):
        for i in xrange(iterations):
            self.evolve()
