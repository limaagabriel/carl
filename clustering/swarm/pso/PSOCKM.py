import numpy as np
import random

from optimization.metaheuristics.PSO import PSO
from optimization.problems.ObjectiveFunction import ObjectiveFunction
from optimization.problems.SearchSpaceInitializer import SearchSpaceInitializer
from clustering.evaluation.ClusteringUtil import ClusteringUtil
from clustering.evaluation.ClusteringUtil import ClusteringSolution

from clustering.debugger.ClusteringDebugger import PartitionalClusteringDebugger

from clustering.traditional.KMeans import KMeans


class PSOCKM_SearchSpaceInitializer(SearchSpaceInitializer):
    def __init__(self, n_clusters, dataset):
        super(PSOCKM_SearchSpaceInitializer, self).__init__()
        self.dataset = dataset
        self.n_clusters = n_clusters

    def sample(self, objective_function, n):

        if self.n_clusters > self.dataset.shape[0]:
            raise Exception("Number of clusters is greater than dataset!")

        candidates = []

        # shuffle_dataset = shuffle(self.dataset, random_state=5)

        idx_shuffle_dataset = np.random.permutation(self.dataset.shape[0])
        shuffle_dataset = np.zeros(self.dataset.shape)

        for i in range(len(idx_shuffle_dataset)):
            shuffle_dataset[i] = self.dataset[idx_shuffle_dataset[i]]

        for i in range(n):
            s = shuffle_dataset.shape[0]

            count = 0
            r = np.random.randint(0, s)
            indices = []
            indices.append(r)
            while (count < (self.n_clusters - 1)):
                r = np.random.randint(0, s)
                a = indices[count]
                if (np.linalg.norm(shuffle_dataset[r, :] - shuffle_dataset[a, :]) < 10e-6):
                    pass
                else:
                    count = count + 1
                    indices.append(r)

            r = indices
            vector = shuffle_dataset[r, :]
            temp = self.n_clusters * shuffle_dataset.shape[1]
            vector = np.reshape(vector, temp)
            candidates.append(vector)

        return candidates


class PSOCKMObjectiveFunction(ObjectiveFunction):
    def __init__(self, data, n_clusters, n_attributes):
        super(PSOCKMObjectiveFunction, self).__init__('PSOC', (n_clusters * n_attributes), 0.0, 1.0)
        self.n_clusters = n_clusters
        self.n_attributes = n_attributes
        self.data = data

    def evaluate(self, x):
        centroids = x.reshape((self.n_clusters, self.n_attributes))
        clusters = {}

        for k in range(self.n_clusters):
            clusters[k] = []

        for xi in self.data:
            # dist = [(np.linalg.norm(xi - centroids[c])**2) for c in range(len(centroids))]
            dist = [ClusteringUtil.squared_euclidean_dist(xi, centroids[c]) for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return ClusteringUtil.sse(centroids, clusters)


class PSOCKM(object):
    def __init__(self, n_clusters=2, swarm_size=100, n_iter=500, w=0.72, c1=1.49, c2=1.49):
        self.n_clusters = n_clusters
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.isTrained = False

        self.up_w = w
        self.lb_w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = 0.5
        self.pso = None

        # self.debugger = PartitionalClusteringDebugger()

    def fit(self, data):
        self.isTrained = True

        # self.debugger = PartitionalClusteringDebugger()

        self.n_attributes = data.shape[1]

        initializer = PSOCKM_SearchSpaceInitializer(self.n_clusters, data)

        # initializer = UniformSSInitializer()

        self.pso = PSO(PSOCKMObjectiveFunction(data, self.n_clusters, self.n_attributes),
                       search_space_initializer=initializer,
                       swarm_size=self.swarm_size,
                       n_iter=self.n_iter,
                       lb_w=self.lb_w, c1=self.c1, c2=self.c2, v_max=self.v_max)

        self.pso.optimize()

        self.centroids = {}
        raw_centroids = self.pso.gbest_particle.pos.reshape((self.n_clusters, self.n_attributes))

        for c in range(len(raw_centroids)):
            self.centroids[c] = raw_centroids[c]

        kmeans = KMeans(n_clusters=self.n_clusters, n_iter=1000)
        kmeans.set_initial_solution(dict(self.centroids))
        kmeans.fit(data)
        self.centroids = dict(kmeans.centroids)

        self.solution = ClusteringSolution(centroids=self.centroids, dataset=data)

        return self.n_iter

    def reset(self):
        self.isTrained = False

    def predict(self, x):
        if self.isTrained:
            dist = []
            for c in self.centroids:
                temp_centroid = np.tile(self.centroids[c], (np.shape(x)[0], 1))
                diff = (x - self.centroids[c])
                power2 = (x - self.centroids[c]) ** 2
                sum = np.sum((x - self.centroids[c]) ** 2, axis=1)
                dist.append(sum)
            labels = np.argmin(dist, axis=0)
            return np.array(labels).T
            # return np.argmin(np.array(class_).T, axis=1)

        raise Exception("NonTrainedModelException: You must fit data first!")
