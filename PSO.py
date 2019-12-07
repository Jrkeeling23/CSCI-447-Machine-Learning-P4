from NeuralNetwork import NeuralNetwork
import random
import numpy as np

class PSO:
    def __init__(self, data_instance, no_of_particles, no_of_nodes, no_of_layers):
        self.data_instance = data_instance
        self.particles = []
        for i in range(no_of_particles):
            self.particles.append(NeuralNetwork(self.data_instance))
            self.particles[i].make_layers(no_of_layers, no_of_nodes)
        self.group_best = None

    def move_them(self, c1, c2, omega):
        for particle in self.particles:
            fit = 0
            for index, row in self.data_instance.train_df.iterrows():
                acc = self.accuracy(particle.sigmoid(row.drop(self.data_instance.label_col)), particle.output_vector, row[self.data_instance.label_col])
                fit += acc
                # print(acc)
                # print(fit)
            particle.fitness = fit/self.data_instance.train_df.shape[0]
            # print(particle.vectorize())
            # print(self.data_instance.train_df.shape)
            print(particle.fitness)

            if particle.personal_best == None:
                particle.personal_best = [particle.vectorize(), particle.fitness]
            elif particle.personal_best[1]<particle.fitness:
                particle.personal_best = [particle.vectorize(), particle.fitness]

            if self.group_best == None:
                self.group_best = [particle.vectorize(), particle.fitness]
            elif self.group_best[1]<particle.fitness:
                self.group_best = [particle.vectorize(), particle.fitness]

        for particle in self.particles:
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            vector = particle.vectorize()
            var1 = c1*r1
            var2 = c2*r2
            if particle.velocity == None:
                particle.velocity = [0]*len(vector)
            for loc in range(len(vector)):
                particle.velocity[loc] = omega*particle.velocity[loc]+var1*(particle.personal_best[0][loc]-vector[loc])+var2*(self.group_best[0][loc]-vector[loc])
            for i in range(len(vector)):
                vector[i] = vector[i]+particle.velocity[i]
            particle.layers = particle.networkize(vector)





    def accuracy(self, pred_output, output_layer, actual):
        high_value = 0
        print(pred_output)
        print(output_layer)
        print(actual)
        for i in range(len(output_layer)):
            if output_layer[i] == actual:
                high_value = i
        prediction = 0
        for f in range(len(pred_output)):
            if pred_output[f]>prediction:
                prediction = f
        print(prediction)
        print(high_value)
        if prediction != high_value:
            return 1
        else:
            return 0
