import numpy as np
from clustering.debugger.ClusteringDebugger import PartitionalClusteringDebugger

class ClusteringSolution(object):
    def __init__(self,centroids,dataset):
        self.centroids = centroids

        self.clusters = ClusteringUtil.assignment_clusters(centroids=centroids,dataset=dataset)

        self.number_of_effective_clusters = 0

        for c in range(len(self.centroids)):
            if len(self.clusters[c]) > 0:
                self.number_of_effective_clusters = self.number_of_effective_clusters + 1






class ClusteringUtil(object):

    @staticmethod
    def assignment_clusters(centroids, dataset):
        clusters = {}
        for c in centroids:
            clusters[c] = []

        for xi in dataset:
            dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        return clusters

    @staticmethod
    def quantization_error(centroids, clusters):
        global_intra_cluster_sum = 0.0

        number_of_effective_clusters = 0

        for c in range(len(centroids)):
            partial_intra_cluster_sum = 0.0
            normalized_partial_intra_cluster_sum = 0.0

            if len(clusters[c]) > 0:
                number_of_effective_clusters = number_of_effective_clusters + 1
                for point in clusters[c]:
                    partial_intra_cluster_sum += np.linalg.norm(point - centroids[c])

                normalized_partial_intra_cluster_sum = (partial_intra_cluster_sum / float(len(clusters[c])))

            global_intra_cluster_sum += normalized_partial_intra_cluster_sum

        if number_of_effective_clusters > 0:
            normalized_global_intra_cluster_sum = (global_intra_cluster_sum / float(number_of_effective_clusters))
            return normalized_global_intra_cluster_sum
        else:
            raise Exception("Arithmetic Error: Division By Error!")



    @staticmethod
    def create_new_debugger(debugger_optimizer, n_clusters, n_attributes):
        particional_clustering_debugger = PartitionalClusteringDebugger()
        particional_clustering_debugger.rep_iteration = debugger_optimizer.repository_iteration
        particional_clustering_debugger.rep_best_cost = debugger_optimizer.best_cost_iteration
        best_solution_evolution = debugger_optimizer.best_solution_iteration

        for i in range(len(best_solution_evolution)):
            centroid = best_solution_evolution[i].reshape((n_clusters, n_attributes))
            particional_clustering_debugger.rep_best_centroids.append(centroid)

        return particional_clustering_debugger


    @staticmethod
    def sse(centroids, clusters):
        global_intra_cluster_sum = 0.0

        for c in range(len(centroids)):
            partial_intra_cluster_sum = 0.0

            if len(clusters[c]) > 0:
                for point in clusters[c]:
                    partial_intra_cluster_sum += (ClusteringUtil.squared_euclidean_dist(point,centroids[c]))

            global_intra_cluster_sum += partial_intra_cluster_sum

        return global_intra_cluster_sum

    @staticmethod
    def squared_euclidean_dist(u,v):
        sed = (((u-v)**2)).sum()
        return sed