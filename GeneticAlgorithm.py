import numpy as np
import random
from ex1 import NN
import copy
import multiprocessing
import time

class Chromosome:
    def __init__(self, length):
        # self.weights = np.random.uniform(0, 1, length)
        eps_w1 = np.sqrt(6.0 / (784+128))
        eps_w2 = np.sqrt(6.0 / (128+64))
        eps_w3 = np.sqrt(6.0 / (64+10))
        eps_b1 = np.sqrt(6.0 / (784 + 1))
        eps_b2 = np.sqrt(6.0 / (128 + 1))
        eps_b3 = np.sqrt(6.0 / (64 + 1))
        self.weights = []
        self.weights.extend(np.random.uniform(-eps_w1, eps_w1, 784 * 128).tolist())
        self.weights.extend(np.random.uniform(-eps_b1, eps_b1, 128).tolist())
        self.weights.extend(np.random.uniform(-eps_w2, eps_w2, 128 * 64).tolist())
        self.weights.extend(np.random.uniform(-eps_b2, eps_b2, 64).tolist())
        self.weights.extend(np.random.uniform(-eps_w3, eps_w3, 64 * 10).tolist())
        self.weights.extend(np.random.uniform(-eps_b3, eps_b3, 10).tolist())


    def mutate(self):
        """
        Randomly mutate one part of the network.
        """
        self.weights += np.random.normal(0.0, 0.1, len(self.weights))

    def fitness(self, network, test):
        # a = time.time()
        w1_values = self.weights[
                     0:
                     network.hlayer1_size * 784]
        network.b1 = self.weights[
                     network.hlayer1_size * 784:
                     network.hlayer1_size * 784 + network.hlayer1_size]
        w2_values = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size :
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size]
        network.b2 = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size:
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
                     network.hlayer2_size]
        w3_values = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
                     network.hlayer2_size:
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
                     network.hlayer2_size + network.hlayer2_size*10]
        network.b3 = self.weights[
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
                     network.hlayer2_size + network.hlayer2_size * 10:
                     network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
                     network.hlayer2_size + network.hlayer2_size * 10 + 10]

        network.W1 = np.reshape(w1_values, (network.hlayer1_size, 784))
        network.W2 = np.reshape(w2_values, (network.hlayer2_size, network.hlayer1_size))
        network.W3 = np.reshape(w3_values, (10, network.hlayer2_size))

        # b = time.time()
        val = network.get_acc_and_loss(test[0], test[1])
        # c = time.time()

        # print str(b-a), " - ", str(c-b)

        return val


# def g_fitness(chrom, network, test):
#         w1_values = chrom.weights[
#                      0:
#                      network.hlayer1_size * 784]
#         network.b1 = chrom.weights[
#                      network.hlayer1_size * 784:
#                      network.hlayer1_size * 784 + network.hlayer1_size]
#         w2_values = chrom.weights[
#                      network.hlayer1_size * 784 + network.hlayer1_size :
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size]
#         network.b2 = chrom.weights[
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size:
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
#                      network.hlayer2_size]
#         w3_values = chrom.weights[
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
#                      network.hlayer2_size:
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
#                      network.hlayer2_size + network.hlayer2_size*10]
#         network.b3 = chrom.weights[
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
#                      network.hlayer2_size + network.hlayer2_size * 10:
#                      network.hlayer1_size * 784 + network.hlayer1_size + network.hlayer1_size * network.hlayer2_size +
#                      network.hlayer2_size + network.hlayer2_size * 10 + 10]
#
#         network.W1 = np.reshape(w1_values, (network.hlayer1_size, 784))
#         network.W2 = np.reshape(w2_values, (network.hlayer2_size, network.hlayer1_size))
#         network.W3 = np.reshape(w3_values, (10, network.hlayer2_size))
#
#         return network.get_acc_and_loss(test[0], test[1])


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
        if type == "n_points":
            for _ in range(1):
                c = Chromosome(self.chromosome_length)
                for i in xrange(self.chromosome_length):
                    c.weights[i] = random.choice([father.weights[i], mother.weights[i]])
                children.append(c)

        elif type == "single_point":
            cross_points = random.randint(1, self.chromosome_length-1)
            c1 = Chromosome(self.chromosome_length)
            c1.weights[:cross_points] = mother.weights[:cross_points]
            c1.weights[cross_points:] = father.weights[cross_points:]
            c2 = Chromosome(self.chromosome_length)
            c2.weights[:cross_points] = father.weights[:cross_points]
            c2.weights[cross_points:] = mother.weights[cross_points:]
            children.extend([c1, c2])

        elif type == "n_points_weights":
            # a = time.time()
            father_w1 = np.reshape(father.weights[0: self.inner_network.hlayer1_size * 784], (self.inner_network.hlayer1_size, 784))
            father_b1 = father.weights[self.inner_network.hlayer1_size * 784: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size]
            father_w2 = np.reshape(father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size : self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size], (self.inner_network.hlayer2_size, self.inner_network.hlayer1_size))
            father_b2 = father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size : self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size ]
            father_w3 = np.reshape(father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size + self.inner_network.hlayer2_size * 10],(10 ,self.inner_network.hlayer2_size))
            father_b3 = father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size + self.inner_network.hlayer2_size * 10: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size + self.inner_network.hlayer2_size * 10 + 10]

            mother_w1 = np.reshape(mother.weights[0: self.inner_network.hlayer1_size * 784], (self.inner_network.hlayer1_size, 784))
            mother_b1 = mother.weights[self.inner_network.hlayer1_size * 784 : self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size]
            mother_w2 = np.reshape(father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size],(self.inner_network.hlayer2_size, self.inner_network.hlayer1_size))
            mother_b2 = mother.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size]
            mother_w3 = np.reshape(father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size + self.inner_network.hlayer2_size * 10],(10, self.inner_network.hlayer2_size))
            mother_b3 = father.weights[self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size + self.inner_network.hlayer2_size * 10: self.inner_network.hlayer1_size * 784 + self.inner_network.hlayer1_size + self.inner_network.hlayer1_size * self.inner_network.hlayer2_size + self.inner_network.hlayer2_size + self.inner_network.hlayer2_size * 10 + 10]

            for _ in range(1):
                c = Chromosome(self.chromosome_length)
                weights = []
                for i in range(0, self.inner_network.hlayer1_size):
                    weights.extend(random.choice([father_w1[i].tolist(), mother_w1[i].tolist()]))
                for i in range(0, self.inner_network.hlayer1_size):
                    weights.append(random.choice([father_b1[i], mother_b1[i]]))
                for i in range(0, self.inner_network.hlayer2_size):
                    weights.extend(random.choice([father_w2[i].tolist(), mother_w2[i].tolist()]))
                for i in range(0, self.inner_network.hlayer2_size):
                    weights.append(random.choice([father_b2[i], mother_b2[i]]))
                for i in range(0, 10):
                    weights.extend(random.choice([father_w3[i].tolist(), mother_w3[i].tolist()]))
                for i in range(0, 10):
                    weights.append(random.choice([father_b3[i], mother_b3[i]]))

                c.weights = weights
                children.append(c)
            # b = time.time()
            # print str(b-a)
        return children

    def evolve(self):
        """Evolve a population of chromosomes.
        Args:
            pop (list): A list of network parameters
        """

        # select randomly 100 examples from the test collection. select the same 100 sample for each chromosome check
        # in generation
        # random.seed(1)
        rnd_indices = random.sample(range(len(self.train[1])), 1000)
        train_x = np.asarray([self.train[0][i] for i in rnd_indices])
        train_y = np.asarray([self.train[1][i] for i in rnd_indices])

        # get random activation and derivative functions
        self.inner_network.active_func, self.inner_network.active_func_deriv = random.choice(self.activation_options)

        # Get scores for each network
        ranked = [(chrom.fitness(self.inner_network, [train_x, train_y]), chrom) for chrom in self.population]
        graded = [(r[0][0], r[1]) for r in list(ranked)]

        print "acc|loss: {:^3.2f} | {:^3.2f}".format(np.mean([r[0][0] for r in ranked]), np.mean([r[0][1] for r in ranked])), # + str(np.mean([r[0][0] for r in ranked])) + "/" + str(np.mean([r[0][1] for r in ranked])),

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
            p_parents = select_parents(graded_copy[:-retain_length], 'ranking')

            male = p_parents[0]
            female = p_parents[1]

            # Breed them.
            actions = ["n_points", "n_points_weights", "single_point"]
            # babies = self.crossover(male, female, random.choice(actions))
            babies = self.crossover(male, female, "n_points_weights")

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
