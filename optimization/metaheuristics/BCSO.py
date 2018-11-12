import copy
import math
import random
import numpy as np
from operator import truediv

# This code was based on in the following references:
# [1] "Defining a Standard for Particle Swarm Optimization" published in 2007 by Bratton and Kennedy
# [2] "Comparing Inertia Weights and Constriction Factors in Particle Swarm Optimization"


# The particle is initialized in a invalid state
class Cat(object):
    def __init__(self, dim):
        self.pos = np.random.choice([0, 1], size=(dim,))
        self.speed = [0.0 for _ in range(dim)]
        self.cost = np.nan
        self.is_seeking = False
        self.prob = 0.0


class BCSO(object):
    def __init__(self, objective_function, swarm_size=50, n_iter=1000, w=0.4, c1=2.05, smp=3, cdc=0.2, pmo=0.2, mr=0.5):

        self.optimum_cost_tracking_eval = []

        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf

        self.swarm_size = swarm_size
        self.n_iter = n_iter

        # Variables that store of state
        self.optimum_cost_tracking_iter = []  # tracking optimum cost
        self.swarm = []
        # gbest of the swarm
        self.best_cat = Cat(self.dim)
        self.best_cat.cost = -np.inf

        self.mr = mr
        self.smp = smp
        self.cdc = cdc
        self.pmo = pmo
        self.pmo = pmo

        # Static parameters of the PSO
        self.w = w
        self.c1 = c1
        self.v_max = 50  # Based on paper [2]
        self.v_min = -50  # Based on paper [2]

    def __init_swarm(self):
        self.best_cat = Cat(self.dim)
        self.best_cat.cost = -np.inf

        for i in range(self.swarm_size):
            cat = Cat(self.dim)
            cat.cost = self.objective_function.evaluate(cat.pos)
            cat.prob = 1.0 / self.swarm_size
            self.optimum_cost_tracking_eval.append(self.best_cat.cost)
            if self.best_cat.cost > cat.cost:
                self.best_cat = copy.deepcopy(cat)
            self.optimum_cost_tracking_eval.append(self.best_cat.cost)
            self.swarm.append(cat)
        self.optimum_cost_tracking_iter.append(self.best_cat.cost)

    def _init_bcso(self):
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def update_best_cat(self):
        for cat in self.swarm:
            if cat.cost >= self.best_cat.cost:
                self.best_cat = copy.deepcopy(cat)

    @staticmethod
    def roulette_wheel(swarm):
        k = range(len(swarm))
        r = np.random.uniform()
        for i in k:
            if r > swarm[i].prob:
                return i
        else:
            return np.random.choice(k)

    def mutate(self, cat):
        selected_dim = random.sample(range(0, self.dim), int(self.cdc * self.dim))
        for d in selected_dim:
            if np.random.uniform() <= self.pmo:
                cat.pos[d] = 1 - cat.pos[d]

    def create_copies(self, cat):
        copies = []
        for n in range(self.smp - 1):
            copies.append(copy.deepcopy(cat))
        return copies

    def seeking(self):
        for cat in range(self.swarm_size):
            if self.swarm[cat].is_seeking:
                copies = self.create_copies(self.swarm[cat])
                for c in copies:
                    self.mutate(c)
                    c.cost = self.objective_function.evaluate(c.pos)
                self.calculate_probabilities(copies)
                selected_cat = self.roulette_wheel(copies)
                self.swarm[cat] = copy.deepcopy(copies[selected_cat])

    @staticmethod
    def calculate_probabilities(swarm_c):
        max_fit = -np.inf
        min_fit = np.inf
        for cat in swarm_c:
            if cat.cost >= max_fit:
                max_fit = cat.cost
            if cat.cost <= min_fit:
                min_fit = cat.cost

        for cat in swarm_c:
            if max_fit != min_fit:
                cat.prob = abs(truediv((max_fit - cat.cost), (max_fit - min_fit)))
            else:
                cat.prob = truediv(1, len(swarm_c))

    def tracing(self):
        for cat in self.swarm:
            if not cat.is_seeking:
                for d in range(self.dim):
                    if cat.pos[d] == 0:
                        if self.best_cat.pos[d] == 0:
                            cat.speed[d] += self.w * cat.speed[d] + random.random() * self.c1
                        else:
                            cat.speed[d] += self.w * cat.speed[d] - random.random() * self.c1
                    else:
                        if self.best_cat.pos[d] == 1:
                            cat.speed[d] += self.w * cat.speed[d] + random.random() * self.c1
                        else:
                            cat.speed[d] += self.w * cat.speed[d] - random.random() * self.c1

                    if cat.speed[d] > self.v_max:
                        cat.speed[d] = self.v_max
                    elif cat.speed[d] < self.v_min:
                        cat.speed[d] = self.v_min

                    # mohamadeen
                    sigmoid_speed = truediv(1, 1 + math.pow(math.e, -cat.speed[d]))
                    if random.random() < sigmoid_speed:
                        cat.pos[d] = self.best_cat.pos[d]


    @staticmethod
    def roulette_wheel(swarm):
        k = range(len(swarm))
        r = np.random.uniform()
        for i in k:
            if r > swarm[i].prob:
                return i
        else:
            return np.random.choice(k)

    def random_choice_mode(self):
        choice = np.arange(0, self.swarm_size)
        random.shuffle(choice)
        for cat in self.swarm:
            cat.is_seeking = False
        for p in range(int(self.mr * self.swarm_size)):
            self.swarm[choice[p]].is_seeking = True

    def optimize(self):
        self._init_bcso()
        self.__init_swarm()

        for i in range(self.n_iter):
            self.random_choice_mode()
            self.seeking()
            self.tracing()
            self.update_best_cat()

            self.optimum_cost_tracking_iter.append(self.best_cat.cost)
            # print('LOG : Iteration {} --- GBEST: {}'.format(i, self.best_cat.cost))
