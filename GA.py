import pandas as pd
import numpy as np
import random
import math
from NeuralNetwork import  NeuralNetwork
from NeuralNetwork import  NetworkClient
from Data import Data

# class to represent chromosomes
class Chromosome:
    # net_vector represents the veector that this chromosome holds
    # layers represents the layers the NN contains, used to turn back into a NN form
    # network the NN this holds, used to turn into vector or into layer form
    # fitness represents the fitness value of that vector

    def __init__(self, net_vector, network, layers, fitness = 0,):

        self.net_vector = net_vector
        self.network = network
        self.fitness = fitness
        self.layers = layers



# class for genetic algorithm

class GA:
    # t_size is the size of a tournement
    # cross over prob is the probability used to select a gene from parent 1
    # mutation rate is the chance of mutation
    # pop size is size of population
    # population is list of chromosomes
    # data is a data instance
    # layers is # of layers for NN's
    # nodes is # of nodes in NN's
    def __init__(self,pop_size, t_size, data, layers =2, nodes = 5, crossover_prob = .5, mutation_rate = .01):
        self.t_size = t_size
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.population = []
        self.data = data
        self.layers = layers
        self.nodes = nodes

    # calculate fitness given some target chromosome
    def CalcFitness(self, target):
        # TODO:  Take a vector from the target, turn it into vector and calculate fitness with testing funcs
         test = None

    # init the population of the GA
    def init_pop(self):
        data = self.data
        # counter for below loop
        counter = 0
        print("Creating Population")
        while counter < self.pop_size:
            # create a new random network, create a new chromosome with it
            network = NeuralNetwork(data_instance=data)
            layers, outputs = network.make_layers(self.layers, self.nodes)
            net_vector = network.vectorize(layers)
            newPop = Chromosome(net_vector, network, layers)
            self.population.append(newPop)
            # calculate fitness of each pop member
            newPop.fitness = self.CalcFitness(network)


            counter+=1
        print("Done Creating Population")

    # function to train network using GA,
    def run_GA(self):
       # setting up the population
        self.init_pop()

        print("WIP")