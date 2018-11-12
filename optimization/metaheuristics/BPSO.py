from __future__ import division

import copy
import numpy as np

from functools import partial


class BinaryParticle(object):

    BINARY_BASE = 2

    def __init__(self, dim, maximize=True):
        self.dim = dim
        self.pos = BinaryParticle.__initialize_position(dim)
        self.speed = np.zeros((1, dim), dtype=np.float32).reshape(dim)
        self.cost = -np.inf if maximize else np.inf
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost

    def update_components(self, w, c1, c2, v_max, gbest):
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        self.speed = w * self.speed + c1 * r1 * (self.pbest_pos - self.pos) + \
                     c2 * r2 * (gbest.pos - self.pos)
        self.restrict_vmax(v_max)
        self.update_pos()

    def restrict_vmax(self, v_max):
        self.speed[self.speed > v_max] = v_max
        self.speed[self.speed < -v_max] = -v_max

    def update_pos(self):
        probs = map(BinaryParticle.__sgm, self.speed)
        prob = np.random.random(self.dim)
        self.pos[probs > prob] = 1
        self.pos[probs < prob] = 0

    @staticmethod
    def __sgm(v):
        return 1 / (1 + np.exp(-v))

    @staticmethod
    def __initialize_position(dim):
        return np.random.randint(BinaryParticle.BINARY_BASE, size=dim)


class BPSO(object):

    def __init__(self, fitness, pop_size=1000, max_iter=5000, lb_w=0.4, up_w=0.9, c1=2.05, c2=2.05,
                 v_max=100000, maximize=True):
        self.c1 = c1
        self.c2 = c2
        self.w = up_w
        self.lb_w = lb_w
        self.up_w = up_w
        self.v_max = min(v_max, 100000)
        self.dim = fitness.dim
        self.fitness = fitness
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.maximize = maximize
        self.op = max if maximize else min

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []
        self.optimal_solution = None

    def optimize(self):
        self.__init_swarm()
        self.__select_best_particle()

        for itr in xrange(self.max_iter):
            self.__update_components()
            self.__evaluate_swarm()
            self.__select_best_particle()
            self.__update_inertia_weight(itr)
            self.optimum_cost_tracking_iter.append(self.optimal_solution.cost)
            # print('LOG : Iteration {} --- GBEST: {}'.format(itr, self.optimal_solution.cost))

    def __init_swarm(self):
        self.w = self.up_w
        self.swarm = []
        self.optimum_cost_tracking_iter = []
        for _ in xrange(self.pop_size):
            self.swarm.append(BinaryParticle(self.dim, self.maximize))

    def __evaluate_swarm(self):
        evaluate = partial(BPSO.__evaluate, self.fitness, self.op)
        map(evaluate, self.swarm)

    def __select_best_particle(self):
        current_optimal = copy.deepcopy(self.op(self.swarm, key=lambda p: p.cost))
        if not self.optimal_solution:
            self.optimal_solution = current_optimal
            return

        if current_optimal.cost > self.optimal_solution.cost:
            self.optimal_solution = current_optimal

    def __update_components(self):
        update = partial(BPSO.__update_swarm_components, self.up_w, self.c1, self.c2,
                         self.v_max, self.optimal_solution)
        map(update, self.swarm)

    def __update_inertia_weight(self, itr):
        self.w = self.up_w - (float(itr) / self.max_iter) * (self.up_w - self.lb_w)

    @staticmethod
    def __evaluate(fitness, op, particle):
        particle.cost = fitness.evaluate(particle.pos)
        particle.pbest_cost = op(particle.cost, particle.pbest_cost)
        if particle.pbest_cost == particle.cost:
            particle.pbest_pos = particle.pos

    @staticmethod
    def __update_swarm_components(w, c1, c2, vmax, gbest, particle):
        particle.update_components(w, c1, c2, vmax, gbest)
