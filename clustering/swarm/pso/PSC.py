import numpy as np
from clustering.evaluation.ClusteringUtil import ClusteringSolution

from clustering.debugger.ClusteringDebugger import PartitionalClusteringDebugger

from clustering.evaluation.ClusteringUtil import ClusteringUtil


# This code was based on the following in the following references:
# [1]  "Data Clustering with Particle Swarms"

class Particle(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.speed = [nan for _ in range(dim)]
        self.dist = np.inf

        self.pbest = self.pos
        self.pbest_dist = np.inf


class PSC(object):
    def __init__(self, swarm_size=100, n_iter=500,
                 w=0.95, c1=2.05, c2=2.05, c3=1.0, c4=1.0, v_max=0.001):
        self.isTrained = False
        self.minf = 0.0
        self.maxf = 1.0
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.w = w
        self.w_damp = 0.95
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.min_c1 = 0.1
        self.min_c2 = 0.1
        self.min_c3 = 0.005
        self.min_c4 = 0.1
        self.v_max = v_max
        self.dim = None
        self.N = None

        self.swarm = []
        self.gbest = []

        self.centroids = {}
        self.solution = None

    def init_swarm(self, N, dim):
        self.swarm = []
        self.gbest = []

        for i in range(self.swarm_size):
            particle = Particle(dim)
            particle.pos = np.random.uniform(self.minf, self.maxf, dim)
            particle.pbest = particle.pos
            particle.speed = np.random.uniform(-1 * self.v_max, self.v_max, dim)
            self.swarm.append(particle)

        for i in range(N):
            particle = Particle(dim)
            self.gbest.append(particle)

    def fit(self, dataset):
        self.isTrained = True
        self.dim = dataset.shape[1]
        self.N = dataset.shape[0]

        self.init_swarm(self.N, self.dim)

        for itr in range(self.n_iter):

            win = np.zeros(len(self.swarm))

            for i in range(len(dataset)):

                dist = np.zeros(len(self.swarm))
                for j in range(len(self.swarm)):
                    dist[j] = np.linalg.norm(self.swarm[j].pos - dataset[i])
                    self.swarm[j].dist = dist[j]

                idx_min_dist = int(dist.argmin())
                win[idx_min_dist] += 1
                if self.swarm[idx_min_dist].dist < self.swarm[idx_min_dist].pbest_dist:
                    self.swarm[idx_min_dist].pbest = self.swarm[idx_min_dist].pos
                    # print(np.sum(self.swarm[idx_min_dist].pos))
                    self.swarm[idx_min_dist].pbest_dist = self.swarm[idx_min_dist].dist

                if self.swarm[idx_min_dist].dist < self.gbest[i].dist:
                    self.gbest[i].pos = self.swarm[idx_min_dist].pos
                    self.gbest[i].dist = self.swarm[idx_min_dist].dist

                phi1 = self.min_c1 + (self.c1 - self.min_c1) * np.random.random(len(self.swarm[idx_min_dist].pos))
                phi2 = self.min_c2 + (self.c2 - self.min_c2) * np.random.random(len(self.swarm[idx_min_dist].pos))
                phi3 = self.min_c3 + (self.c3 - self.min_c3) * np.random.random(len(self.swarm[idx_min_dist].pos))

                self.swarm[idx_min_dist].speed = self.w * self.swarm[idx_min_dist].speed \
                                                 + phi1 * (
                    self.swarm[idx_min_dist].pbest - self.swarm[idx_min_dist].pos) \
                                                 + phi2 * (
                    self.gbest[i].pos - self.swarm[idx_min_dist].pos) \
                                                 + phi3 * (dataset[i] - self.swarm[idx_min_dist].pos)

                vel = self.swarm[idx_min_dist].speed
                self.swarm[idx_min_dist].speed = np.sign(vel) * np.minimum(
                    np.absolute(vel), np.ones(self.dim) * self.v_max)

                self.swarm[idx_min_dist].pos = self.swarm[idx_min_dist].pos + self.swarm[idx_min_dist].speed

                if (self.swarm[idx_min_dist].pos < self.minf).any() or (self.swarm[idx_min_dist].pos > self.maxf).any():
                    self.swarm[idx_min_dist].speed[self.swarm[idx_min_dist].pos < self.minf] = -1 * self.swarm[
                        idx_min_dist].speed[self.swarm[idx_min_dist].pos < self.minf]

                    self.swarm[idx_min_dist].speed[self.swarm[idx_min_dist].pos > self.maxf] = -1 * self.swarm[
                        idx_min_dist].speed[self.swarm[idx_min_dist].pos > self.maxf]

                    self.swarm[idx_min_dist].pos[self.swarm[idx_min_dist].pos > self.maxf] = self.maxf
                    self.swarm[idx_min_dist].pos[self.swarm[idx_min_dist].pos < self.minf] = self.minf

            p_win_most = self.swarm[np.argmax(win)]
            for i in range(len(self.swarm)):
                if win[i] == 0:
                    phi4 = self.min_c4 + (self.c4 - self.min_c4) * np.random.random(self.dim)

                    p = self.swarm[i]
                    p.speed = self.w * p.speed + phi4 * (p_win_most.pos - p.pos)

                    p.speed = np.sign(p.speed) * np.minimum(
                        np.absolute(p.speed), np.ones(self.dim) * self.v_max)

                    p.pos = p.pos + p.speed

                    if (p.pos < self.minf).any() or (p.pos > self.maxf).any():
                        p.speed[p.pos < self.minf] = -1 * p.speed[p.pos < self.minf]
                        p.speed[p.pos > self.maxf] = -1 * p.speed[p.pos > self.maxf]
                        p.pos[p.pos > self.maxf] = self.maxf
                        p.pos[p.pos < self.minf] = self.minf

            self.w = self.w * self.w_damp

        self.centroids = {}
        for i in range(len(self.swarm)):
            self.centroids[i] = self.swarm[i].pbest

        self.solution = ClusteringSolution(centroids=self.centroids, dataset=dataset)

        return self.n_iter

    def add_entries_debugger(self, curr_iteration, dataset):
        centroids = {}
        for i in range(len(self.swarm)):
            centroids[i] = self.swarm[i].pbest
        clusters = {}
        for k in range(self.swarm_size):
            clusters[k] = []
        for xi in dataset:
            # dist = [(np.linalg.norm(xi - centroids[c])**2) for c in range(len(centroids))]
            dist = [ClusteringUtil.squared_euclidean_dist(xi, centroids[c]) for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)
        ClusteringUtil.sse(centroids, clusters)

    def predict(self, x):
        if self.isTrained:
            if len(x.shape) > 1:
                class_ = []
                for c in self.centroids:
                    class_.append(np.sum((x - self.centroids[c]) ** 2, axis=1))
                return np.argmin(np.array(class_).T, axis=1)
            else:
                dist = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
                class_ = dist.index(min(dist))
                return class_
        else:
            raise Exception("NonTrainedModelException: You must fit data first!")
