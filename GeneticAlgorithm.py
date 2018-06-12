import numpy as np
import random
from ex1 import NN
import copy

class Chromosome:
    def __init__(self, length):
        self.weights = np.random.uniform(0,1,length)

    def mutate(self):
        """
        Randomly mutate one part of the network.
        """
        self.weights += np.random.normal(-0.5, 0.5, len(self.weights))

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

        # select randomly 100 examples from the test collection. select the same 100 sample for each chromosome check
        # in generation
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

    def crossover(self, mother, father, type="single_point"):
        """Make two children as parts of their parents.
        Args:
            mother (chromosome): Network weights
            father (chromosome): Network weights
        """
        children = []
        if type == "n_points":
            for _ in range(2):
                c = Chromosome(self.chromosome_length)
                for i in xrange(self.chromosome_length):
                    c.weights[i] = random.choice([father.weights[i], mother.weights[i]])
                children.append(c)

        elif type == "single_point":
            cross_points = random.randint(1,self.chromosome_length-1)
            c1 = Chromosome(self.chromosome_length)
            c1.weights[:cross_points] = mother.weights[:cross_points]
            c1.weights[cross_points:] = father.weights[cross_points:]
            c2 = Chromosome(self.chromosome_length)
            c2.weights[:cross_points] = father.weights[:cross_points]
            c2.weights[cross_points:] = mother.weights[cross_points:]
            children.extend([c1,c2])

        elif type == "n_points_weights":
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

            for _ in range(2):
                c = Chromosome(self.chromosome_length)
                weights = []
                for i in range(0,self.inner_network.hlayer1_size):
                    weights.extend(random.choice([father_w1[i].tolist(), mother_w1[i].tolist()]))
                for i in range(0,self.inner_network.hlayer1_size):
                    weights.append(random.choice([father_b1[i],mother_b1[i]]))
                for i in range(0,self.inner_network.hlayer2_size):
                    weights.extend(random.choice([father_w2[i].tolist(), mother_w2[i].tolist()]))
                for i in range(0,self.inner_network.hlayer2_size):
                    weights.append(random.choice([father_b2[i],mother_b2[i]]))
                for i in range(0,10):
                    weights.extend(random.choice([father_w3[i].tolist(),mother_w3[i].tolist()]))
                for i in range(0,10):
                    weights.append(random.choice([father_b3[i],mother_b3[i]]))


                c.weights = weights
                children.append(c)

        return children

    def evolve(self):
        """Evolve a population of chromosomes.
        Args:
            pop (list): A list of network parameters
        """

        # Get scores for each network.
        graded = [(chrom.fitness(self.inner_network,self.test), chrom) for chrom in self.population]
        print "averaged: " + str(np.mean([grade[0] for grade in graded])),

        # Sort on the scores.
        graded = [x for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        # print "graded : " + str([g[0] for g in graded])
        print "best: " + str(self.best_chrom[0])
        graded_copy = list(graded)


        # update best entity and update graded
        if graded[0][0] > self.best_chrom[0]: self.best_chrom = graded[0]
        graded = [x[1] for x in graded]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]
        # if self.best_chrom[1] not in parents:
        #     parents.append(graded[0])

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)


        # Now find out how many spots we have left to fill.
        desired_length = len(self.population) - len(parents)

        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            weights = [element[0] for element in graded_copy]
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights] #normalize weights

            pp = np.random.choice([element[1] for element in graded_copy], 2, True, weights)
            male = pp[0]
            female = pp[1]

            # Breed them.
            actions = ["n_points", "n_points_weights", "single_point"]
            babies = self.crossover(male, female, random.choice(actions))
            # babies = self.crossover(male, female, "single_point")

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)

        for individual in children:
            if self.mutate_chance > random.random():
                individual.mutate()

        parents.extend(children)
        self.population = list(parents)

    def run(self, iterations):
        for i in xrange(iterations):
            random.seed(i)
            print str(i)+":",
            self.evolve()
