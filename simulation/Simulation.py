import numpy as np

from src.optimization.metaheuristics.PSO import PSO


class Simulation(object):

    def __init__(self, pathname, objective_function, search_space_initializer, num_runs, num_particles, num_iterations,
                 c1, c2, lb_w, up_w,v_max):
        self.num_runs = num_runs
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1
        self.c2 = c2
        self.lb_w = lb_w
        self.up_w = up_w
        self.v_max = v_max
        self.pathname = pathname

    def run(self, save=True):
        for run in range(self.num_runs):
            opt1 = PSO(self.objective_function, self.search_space_initializer, swarm_size=self.num_particles,
                       n_iter=self.num_iterations,
                       lb_w=self.lb_w, up_w=self.up_w, c1=self.c1, c2=self.c2, v_max=self.v_max)
            opt1.optimize()

            temp_optimum_cost_tracking = opt1.optimum_cost_tracking_iter
            temp_optimum_position_tracking = opt1.gbest_particle.pos

            if run == 0:
                optimum_cost_tracking_foreach_run = np.asarray(temp_optimum_cost_tracking)
                optimum_position_tracking_foreach_run = np.asarray(temp_optimum_position_tracking)
            else:
                optimum_cost_tracking_foreach_run = np.vstack(
                    (optimum_cost_tracking_foreach_run, np.asarray(temp_optimum_cost_tracking)))
                optimum_position_tracking_foreach_run = np.vstack(
                    (optimum_position_tracking_foreach_run, np.asarray(temp_optimum_position_tracking)))

            if save:
                np.savetxt(function.function_name + "_cost.txt", optimum_cost_tracking_foreach_run)
                np.savetxt(function.function_name + "_position.txt", optimum_position_tracking_foreach_run)
