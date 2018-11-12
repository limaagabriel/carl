import numpy as np

from clustering.traditional.KMeans import KMeans
from optimization.metaheuristics.PSO import PSO
from optimization.problems.ObjectiveFunction import ObjectiveFunction
from optimization.problems.SearchSpaceInitializer import SearchSpaceInitializer
from optimization.problems.SearchSpaceInitializer import UniformSSInitializer

from clustering.swarm.pso.PSOC import PSOC_SearchSpaceInitializer
from clustering.evaluation.ClusteringUtil import ClusteringUtil
from clustering.evaluation.ClusteringUtil import ClusteringSolution

from clustering.debugger.ClusteringDebugger import PartitionalClusteringDebugger


class KMPSOC_SearchSpaceInitializer(SearchSpaceInitializer):
    def __init__(self, n_clusters, dataset, candidate):
        super(KMPSOC_SearchSpaceInitializer, self).__init__()
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.candidate = candidate

    def sample(self, objective_function, n):
        initializer = PSOC_SearchSpaceInitializer(self.n_clusters, self.dataset)

        x = initializer.sample(objective_function, n - 1)

        x = np.vstack((x, [self.candidate]))

        return x


class KMPSOC_ObjectiveFunction(ObjectiveFunction):
    def __init__(self, data, n_clusters, n_attributes):
        super(KMPSOC_ObjectiveFunction, self).__init__('KMPSOC', (n_clusters * n_attributes), 0.0, 1.0)
        self.n_clusters = n_clusters
        self.n_attributes = n_attributes
        self.data = data

    def evaluate(self, x):
        centroids = x.reshape((self.n_clusters, self.n_attributes))
        clusters = {}

        for k in range(self.n_clusters):
            clusters[k] = []

        for xi in self.data:
            dist = [ClusteringUtil.squared_euclidean_dist(xi, centroids[c]) for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return ClusteringUtil.sse(centroids, clusters)


class KMPSOC(object):
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

        self.debugger = None

    def fit(self, data):
        self.isTrained = True
        self.n_attributes = data.shape[1]
        kmeans = KMeans(n_clusters=self.n_clusters, n_iter=1000)
        kmeans.fit(data)

        candidate = []
        for k in kmeans.centroids:
            candidate.append(kmeans.centroids[k])
        candidate = np.array(candidate).ravel()

        objective_function = KMPSOC_ObjectiveFunction(data, self.n_clusters, self.n_attributes)
        search_space_initializer = KMPSOC_SearchSpaceInitializer(self.n_clusters, data, candidate=candidate)

        self.pso = PSO(objective_function=objective_function,
                       search_space_initializer=search_space_initializer,
                       swarm_size=self.swarm_size,
                       n_iter=self.n_iter,
                       lb_w=self.lb_w, c1=self.c1, c2=self.c2, v_max=self.v_max)

        self.pso.optimize()

        self.centroids = {}
        raw_centroids = self.pso.gbest_particle.pos.reshape((self.n_clusters, self.n_attributes))

        for centroid in range(len(raw_centroids)):
            self.centroids[centroid] = raw_centroids[centroid]

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
