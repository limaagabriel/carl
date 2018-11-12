import numpy as np


# This code was based on in the following references:
# [1] "On clarifying misconceptions when comparing variants of the Artificial Bee Colony Algorithm by offering a new
# implementation" published in 2015 by Marjan Mernik, Shih-Hsi Liu, Dervis Karaboga and Matej Crepinsek
# [2] "A modified Artificial Bee Colony algorithm for real-parameter optimization" published in 2010 by
# Bahriye Akay and Dervis Karaboga

# This implementation consider #_employed = #_onlookers = #_food_sources = colony_size / 2

class Bee(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = np.inf
        self.fitness = 0.0
        self.prob = 0.0
        self.trials = 0


class ABC(object):
    def __init__(self, objective_function, search_space_initializer, n_iter=1000, colony_size=100, trials_limit=100):

        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer

        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter

        self.best_bee = None
        self.optimum_cost_tracking_iter = []

        self.num_fs = colony_size / 2
        self.trials_limit = trials_limit
        self.food_sources = []

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

    @staticmethod
    def calculate_fitness(cost):
        if cost >= 0:
            result = 1.0 / (1.0 + cost)
        else:
            result = 1.0 + abs(cost)
        return result

    def calculate_probabilities(self):
        sum_fit = 0.0
        for fs in range(self.num_fs):
            sum_fit += self.food_sources[fs].fitness

        for fs in range(self.num_fs):
            self.food_sources[fs].prob = (self.food_sources[fs].fitness / sum_fit)

    def update_best_solution(self):
        for bee in self.food_sources:
            if bee.cost < self.best_bee.cost:
                self.best_bee.pos = bee.pos
                self.best_bee.cost = bee.cost

    def init_bee(self, pos):
        bee = Bee(self.dim)
        bee.pos = pos
        bee.cost = self.objective_function.evaluate(bee.pos)
        self.optimum_cost_tracking_eval.append(self.best_bee.cost)
        bee.fitness = self.calculate_fitness(bee.cost)
        return bee

    def init_colony(self):
        self.best_bee = Bee(self.dim)
        self.best_bee.cost = np.inf

        positions = self.search_space_initializer.sample(self.objective_function, self.num_fs)
        for i in range(self.num_fs):
            bee = self.init_bee(positions[i])
            self.food_sources.append(bee)

            if bee.cost < self.best_bee.cost:
                self.best_bee.pos = bee.pos
                self.best_bee.cost = bee.cost

    def employed_bee_phase(self):
        for bee in range(self.num_fs):
            k = range(self.num_fs)
            k.remove(bee)
            k = np.random.choice(np.array(k))
            j = np.random.choice(range(self.dim))
            phi = np.random.uniform(-1, 1)

            new_pos = np.copy(self.food_sources[bee].pos)
            new_pos[j] = self.food_sources[bee].pos[j] + phi * (self.food_sources[bee].pos[j] - self.food_sources[k].pos[j])

            if new_pos[j] < self.minf:
                new_pos[j] = self.minf
            elif new_pos[j] > self.maxf:
                new_pos[j] = self.maxf
            cost = self.objective_function.evaluate(new_pos)
            self.optimum_cost_tracking_eval.append(self.best_bee.cost)
            fit = self.calculate_fitness(cost)

            if fit > self.food_sources[bee].fitness:
                self.food_sources[bee].pos = new_pos
                self.food_sources[bee].cost = cost
                self.food_sources[bee].fitness = fit
                self.food_sources[bee].trials = 0
            else:
                self.food_sources[bee].trials += 1

    def onlooker_bee_phase(self):
        t = s = 0
        while t < self.num_fs:
            s = (s + 1) % self.num_fs
            r = np.random.uniform()

            if r < self.food_sources[s].prob:
                t += 1

                k = range(self.num_fs)
                k.remove(s)
                k = np.random.choice(np.array(k))
                j = np.random.choice(range(self.dim))
                phi = np.random.uniform(-1, 1)

                new_pos = np.copy(self.food_sources[s].pos)
                new_pos[j] = new_pos[j] + phi * (new_pos[j] - self.food_sources[k].pos[j])

                if new_pos[j] < self.minf:
                    new_pos[j] = self.minf
                elif new_pos[j] > self.maxf:
                    new_pos[j] = self.maxf
                cost = self.objective_function.evaluate(new_pos)
                self.optimum_cost_tracking_eval.append(self.best_bee.cost)
                fit = self.calculate_fitness(cost)

                if fit > self.food_sources[s].fitness:
                    self.food_sources[s].pos = new_pos
                    self.food_sources[s].cost = cost
                    self.food_sources[s].fitness = fit
                    self.food_sources[s].trials = 0
                else:
                    self.food_sources[s].trials += 1

    def get_max_trial(self):
        max_ = 0
        for fs in range(self.num_fs):
            if self.food_sources[fs].trials > self.food_sources[max_].trials:
                max_ = fs
        return max_

    def scout_bee_phase(self):
        max_ = self.get_max_trial()

        if self.food_sources[max_].trials >= self.trials_limit:
            pos = self.search_space_initializer.sample(self.objective_function, 1)
            self.food_sources[max_] = self.init_bee(pos[0])

    def optimize(self):
        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.init_colony()
        self.update_best_solution()

        for i in range(self.n_iter):
            self.employed_bee_phase()
            self.calculate_probabilities()
            self.onlooker_bee_phase()
            self.update_best_solution()
            self.scout_bee_phase()
            self.optimum_cost_tracking_iter.append(self.best_bee.cost)
            # print "Iteration: ", i, " Cost: ", ("%04.03e" % self.best_bee.cost)
