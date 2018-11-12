import numpy as np
import pickle
import copy

# This code was based on in the following references:
# [1] "Defining a Standard for Particle Swarm Optimization" published in 2007 by Bratton and Kennedy
# [2] "Comparing Inertia Weights and Constriction Factors in Particle Swarm Optimization"


# The particle is initialized in a invalid state
class Particle(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.speed = [nan for _ in range(dim)]
        self.cost = np.nan
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost


class PSO(object):
    def __init__(self, objective_function, search_space_initializer, swarm_size=50, n_iter=1000, lb_w=0.4,
                 up_w=0.9, c1=2.05, c2=2.05, v_max=100000, bkp_file=None):

        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer
        self.optimum_cost_tracking_eval = []

        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.bkp_file = bkp_file

        self.swarm_size = swarm_size
        self.n_iter = n_iter

        # Variables that store of state of PSO
        self.optimum_cost_tracking_iter = []  # tracking optimum cost
        self.swarm = []
        # gbest of the swarm
        self.gbest_particle = Particle(self.dim)
        self.gbest_particle.cost = np.inf

        # Adaptative Parameters of PSO and that therefore must be reset every
        # time which the optimize function is called
        self.w = up_w

        # Static parameters of the PSO
        self.up_w = up_w
        self.lb_w = lb_w
        self.c1 = c1
        self.c2 = c2
        self.v_max = min(v_max, 100000)  # Based on paper [2]

    def __init_swarm(self):

        self.gbest_particle = Particle(self.dim)
        self.gbest_particle.cost = np.inf

        x = self.search_space_initializer.sample(self.objective_function, self.swarm_size)

        for i in range(self.swarm_size):
            particle = Particle(self.dim)
            particle.pos = x[i]
            particle.speed = [0.0 for _ in range(self.dim)]
            particle.cost = self.objective_function.evaluate(particle.pos)
            self.optimum_cost_tracking_eval.append(self.gbest_particle.cost)
            particle.pbest_pos = particle.pos
            particle.pbest_cost = particle.cost
            if particle.pbest_cost < self.gbest_particle.cost:
                self.gbest_particle.pos = particle.pbest_pos
                self.gbest_particle.cost = particle.pbest_cost

            self.swarm.append(particle)

        self.optimum_cost_tracking_iter.append(self.gbest_particle.cost)

    # Restart the PSO
    def _init_pso(self):
        self.w = self.up_w
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def optimize(self):
        self._init_pso()
        self.__init_swarm()

        for i in range(self.n_iter):
            # if self.bkp_file:
            #     file_ = file(self.bkp_file, 'wb', 0)
            #     pickle.dump(self, file_)
            #     file_.close()

            for p in self.swarm:
                r1 = np.random.random(len(p.speed))
                r2 = np.random.random(len(p.speed))
                p.speed = self.w * np.array(p.speed) + self.c1 * r1 * (p.pbest_pos - p.pos) + self.c1 * r2 * (
                    self.gbest_particle.pos - p.pos)

                # Limit the velocity of the particle
                p.speed = np.sign(p.speed) * np.minimum(np.absolute(p.speed), np.ones(self.dim) * self.v_max)

                p.pos = p.pos + p.speed

                # Confinement of the particle in the search space
                if (p.pos < self.minf).any() or (p.pos > self.maxf).any():
                    p.speed[p.pos < self.minf] = -1 * p.speed[p.pos < self.minf]
                    p.speed[p.pos > self.maxf] = -1 * p.speed[p.pos > self.maxf]
                    p.pos[p.pos > self.maxf] = self.maxf
                    p.pos[p.pos < self.minf] = self.minf

                p.cost = self.objective_function.evaluate(p.pos)
                self.optimum_cost_tracking_eval.append(self.gbest_particle.cost)

                if p.cost <= p.pbest_cost:
                    p.pbest_pos = copy.copy(p.pos)
                    p.pbest_cost = p.cost

            for p in self.swarm:
                if p.pbest_cost <= self.gbest_particle.cost:
                    self.gbest_particle.pos = p.pbest_pos
                    self.gbest_particle.cost = p.pbest_cost

            self.w = self.up_w - (float(i) / self.n_iter) * (self.up_w - self.lb_w)
            # print "iter: ", i , " Cost: ", ("%04.03e" % self.gbest_particle.cost)
            self.optimum_cost_tracking_iter.append(self.gbest_particle.cost)
