import numpy as np
import copy

# This code was based on in the following references:
# [1] "Firefly Algorithms for Multimodal Optimization" published in 2009 by Xin-She Yang.


class Firefly(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.brightness = 0.4
        self.cost = np.nan


class FA(object):
    def __init__(self, objective_function, search_space_initializer, n_iter, population_size, alpha=0.2, beta=1, gamma=1):
        self.search_space_initializer = search_space_initializer

        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter

        self.population_size = population_size
        self.population = []
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.best_firefly = None

        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def __init_firefly(self, pos):
        firefly = Firefly(self.dim)
        firefly.pos = pos
        firefly.cost = self.objective_function.evaluate(firefly.pos)
        self.optimum_cost_tracking_eval.append(self.best_firefly.cost)
        return firefly

    def __init_population(self):
        self.best_firefly = Firefly(self.dim)
        self.best_firefly.cost = np.inf
        self.population = []

        positions = self.search_space_initializer.sample(self.objective_function, self.population_size)

        for idx in range(self.population_size):
            ff = self.__init_firefly(positions[idx])
            self.population.append(ff)

        self.update_best()
        self.optimum_cost_tracking_iter.append(self.best_firefly.cost)

    def __init_fa(self):
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def update_best(self):
        for firefly in self.population:
            if self.best_firefly.cost > firefly.cost:
                self.best_firefly = copy.copy(firefly)

    def euclidean_distance(self, xi, xj):
        distance = 0
        for d in range(self.dim):
            distance += np.sqrt((xi[d] - xj[d]) ** 2)
        return distance

    def optimize(self):
        self.__init_fa()
        self.__init_population()

        for iter_ in range(self.n_iter):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self.population[j].cost > self.population[i].cost:
                        new_pos = copy.copy(self.population[i].pos)
                        dist = self.euclidean_distance(self.population[j].pos, self.population[i].pos)
                        beta = self.beta * np.exp(-self.gamma*(dist ** 2))
                        for d in range(self.dim):
                            new_pos[d] += beta * (self.population[j].pos[d] - self.population[i].pos[d]) + \
                                          self.alpha * (np.random.uniform(0, 1) - 0.5)
                            if new_pos[d] < self.minf:
                                new_pos[d] = self.minf
                            elif new_pos[d] > self.maxf:
                                new_pos[d] = self.maxf

                        cost = self.objective_function.evaluate(new_pos)
                        self.optimum_cost_tracking_eval.append(self.best_firefly.cost)
                        if cost < self.population[i].cost:
                            self.population[i].cost = cost
                            self.population[i].pos = new_pos
                    else:
                        new_pos = np.zeros((self.dim,), dtype=np.float)
                        for d in range(self.dim):
                            new_pos[d] = (self.population[i].pos[d] * np.random.uniform(0, 1))
                            if new_pos[d] < self.minf:
                                new_pos[d] = self.minf
                            elif new_pos[d] > self.maxf:
                                new_pos[d] = self.maxf

                        cost = self.objective_function.evaluate(new_pos)
                        self.optimum_cost_tracking_eval.append(self.best_firefly.cost)
                        if cost < self.population[i].cost:
                            self.population[i].cost = cost
                            self.population[i].pos = new_pos
            self.update_best()
            self.optimum_cost_tracking_iter.append(self.best_firefly.cost)
            print "Iteration: ", iter_, " Cost: ", ("%04.03e" % self.best_firefly.cost)
