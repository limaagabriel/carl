import copy
import numpy as np

from clustering.evaluation.ClusteringUtil import ClusteringUtil
from clustering.evaluation.ClusteringUtil import ClusteringSolution
from clustering.debugger.ClusteringDebugger import PartitionalClusteringDebugger


class KMeans(object):
    def __init__(self, n_clusters=2, n_iter=300, shuffle=True, tolerance=0.00000001):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.shuffle = shuffle
        self.run = False
        self.candidates = False
        self.debugger = None

    def set_initial_solution(self, centroids):
        self.centroids = dict(centroids)
        self.candidates = True

    def fit(self, x):
        self.run = True

        self.debugger = PartitionalClusteringDebugger()
        self.debugger.rep_iteration = []
        self.debugger.rep_best_cost = []
        self.debugger.best_solution_evolution = []

        if len(x.shape) < 1:
            raise Exception("DataException: Dataset must contain more examples" +
                            "than the required number of clusters!")

        if (not self.candidates):
            self.centroids = {}
            if self.shuffle:
                r = np.random.permutation(x.shape[0])
                for k in range(len(r[:self.n_clusters])):
                    self.centroids[k] = x[r[k]]
            else:
                for k in range(self.n_clusters):
                    self.centroids[k] = x[k]

        for itr in range(self.n_iter):
            self.clusters = {}
            for k in range(self.n_clusters):
                self.clusters[k] = []

            for xi in x:
                dist = [np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
                class_ = dist.index(min(dist))
                self.clusters[class_].append(xi)

            old_centroids = dict(self.centroids)
            for k in self.clusters:
                if (len(self.clusters[k]) > 0):
                    self.centroids[k] = np.average(self.clusters[k], axis=0)

            is_done = True
            for k in self.centroids:
                old_centroid = old_centroids[k]
                centroid = self.centroids[k]
                if (np.linalg.norm(old_centroid - centroid) > self.tolerance):
                    is_done = False


            self.debugger.rep_iteration.append(itr+1)
            sse_ = ClusteringUtil.sse(centroids=self.centroids, clusters=self.clusters)
            self.debugger.rep_best_cost.append(sse_)
            self.debugger.rep_best_centroids.append(copy.copy(self.centroids))

            if is_done:
                self.solution = ClusteringSolution(centroids=self.centroids, dataset=x)
                return itr
        else:
            return self.n_iter

    def predict(self, x):
        if self.run:
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
