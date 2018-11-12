import random

import numpy as np
import copy

# This code was based on in the following references:
# [1] "A Novel Search Algorithm based on Fish School Behavior" published in 2008 by Bastos Filho, Lima Neto,
# Lins, D. O. Nascimento and P. Lima
# [2] "An Enhanced Fish School Search Algorithm" published in 2013 by Bastos Filho and  D. O. Nascimento
from src.simulation.metaheuristics.Utils import swarm_evolution


class Fish(object):
    def __init__(self, dim):
        self.pos = np.random.choice([0, 1], size=(dim,), p=[3. / 4, 1. / 4])
        self.cost = np.nan
        self.delta_cost = np.nan
        self.weight = np.nan


class BFSS(object):
    def __init__(self, objective_function, n_iter, school_size, threshold_individual_initial, threshold_individual_final,
                    threshold_instintive, threshold_volitive_initial, threshold_volitive_final, min_w, w_scale):
        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter
        self.school_size = school_size

        self.thres_individual_init = threshold_individual_initial
        self.thres_individual_final = threshold_individual_final
        self.thres_volitive_init = threshold_volitive_initial
        self.thres_volitive_final = threshold_volitive_final
        self.threshold_instintive = threshold_instintive
        self.curr_thres_individual = self.thres_individual_init * (self.maxf - self.minf)
        self.curr_thres_volitive = self.thres_volitive_init * (self.maxf - self.minf)

        self.min_w = min_w
        self.w_scale = w_scale
        self.prev_weight_school = 0.0
        self.curr_weight_school = 0.0
        self.best_fish = None

        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def __gen_weight(self):
        return self.w_scale / 2.0

    def __init_fss(self):
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def __init_fish(self):
        fish = Fish(self.dim)
        fish.weight = self.__gen_weight()
        fish.cost = self.objective_function.evaluate(fish.pos)
        self.optimum_cost_tracking_eval.append(self.best_fish.cost)
        return fish

    def __init_school(self):
        self.best_fish = Fish(self.dim)
        self.best_fish.cost = -np.inf
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0
        self.school = []

        for idx in range(self.school_size):
            fish = self.__init_fish()
            self.school.append(fish)
            self.curr_weight_school += fish.weight
            if self.best_fish.cost < fish.cost:
                self.best_fish = copy.copy(fish)
        self.prev_weight_school = self.curr_weight_school
        self.optimum_cost_tracking_iter.append(self.best_fish.cost)

    def max_delta_cost(self):
        max_ = -np.inf
        for fish in self.school:
            if max_ < fish.delta_cost:
                max_ = fish.delta_cost
        return max_

    def total_school_weight(self):
        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = 0.0
        for fish in self.school:
            self.curr_weight_school += fish.weight

    def calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=np.float)
        density = 0.0

        for fish in self.school:
            density += fish.weight
            for dim in range(self.dim):
                barycenter[dim] += (fish.pos[dim] * fish.weight)
        for dim in range(self.dim):
            barycenter[dim] = barycenter[dim] / density

        return barycenter

    def update_steps(self, curr_iter):
        self.curr_thres_individual = self.thres_individual_init - (self.thres_individual_init - self.thres_individual_final) * (
                                                                curr_iter / self.n_iter)

        self.curr_thres_volitive = self.thres_volitive_init - (self.thres_volitive_init - self.thres_volitive_final) * (
                                                                    curr_iter / self.n_iter)

    def update_best_fish(self):
        for fish in self.school:
            if self.best_fish.cost < fish.cost:
                self.best_fish = copy.copy(fish)

    def feeding(self):
        for fish in self.school:
            if self.max_delta_cost():
                fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost())
            if fish.weight > self.w_scale:
                fish.weight = self.w_scale
            elif fish.weight < self.min_w:
                fish.weight = self.min_w

    def individual_movement(self):
        for fish in self.school:
            new_pos = copy.deepcopy(fish.pos)
            for dim in range(self.dim):
                u = np.random.random()
                if u < self.curr_thres_individual:
                    new_pos[dim] = int(not new_pos[dim])
            cost = self.objective_function.evaluate(new_pos)
            if cost > fish.cost:
                fish.delta_cost = cost - fish.cost
                fish.cost = cost
                fish.pos = new_pos
            else:
                fish.delta_cost = 0

    def collective_instinctive_movement(self):
        cost_eval_enhanced = np.zeros((self.dim,), dtype=np.float)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.dim):
                cost_eval_enhanced[dim] += (fish.pos[dim] * fish.delta_cost)
        for dim in range(self.dim):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density

        max_i = max(cost_eval_enhanced)
        new_pos = np.zeros((self.dim,), dtype=np.float)
        for dim in range(self.dim):
            if cost_eval_enhanced[dim] >= self.threshold_instintive * max_i:
                new_pos[dim] = 1

        for fish in self.school:
            for dim in range(self.dim):
                if fish.pos[dim] != new_pos[dim]:
                    fish.pos[dim] = new_pos[dim]
                    break

    def collective_volitive_movement(self):
        self.total_school_weight()
        barycenter = self.calculate_barycenter()

        max_i = max(barycenter)
        bin_baricenter = np.zeros((self.dim,), dtype=np.float)

        for dim in range(self.dim):
            if barycenter[dim] >= self.curr_thres_volitive * max_i:
                bin_baricenter[dim] = 1

        for fish in self.school:
            for dim in range(self.dim):
                if fish.pos[dim] != bin_baricenter[dim]:
                    fish.pos[dim] = bin_baricenter[dim]
                    break

        for fish in self.school:
            if self.curr_weight_school > self.prev_weight_school:
                for dim in range(self.dim):
                    if fish.pos[dim] != bin_baricenter[dim]:
                        fish.pos[dim] = int(not (fish.pos[dim]))
                        break
            else:
                for dim in range(self.dim):
                    if fish.pos[dim] == bin_baricenter[dim]:
                        fish.pos[dim] = int(not (fish.pos[dim]))
                        break

            fish.cost = self.objective_function.evaluate(fish.pos)
            self.optimum_cost_tracking_eval.append(self.best_fish.cost)

    def optimize(self):
        self.__init_fss()
        self.__init_school()

        for i in range(self.n_iter):
            self.individual_movement()
            self.update_best_fish()
            self.feeding()
            self.collective_instinctive_movement()
            self.collective_volitive_movement()
            self.update_steps(i)
            self.update_best_fish()
            self.optimum_cost_tracking_iter.append(self.best_fish.cost)
            # print('LOG : Iteration {} --- GBEST: {}'.format(i, self.best_fish.cost))
