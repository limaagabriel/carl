

class PartitionalClusteringDebugger(object):
    def __init__(self):
        self.rep_best_centroids = []
        self.rep_best_cost = []
        self.rep_iteration = []

    def add_best_cluster(self, curr_iteration, curr_best_centroids, curr_best_cost):
        self.rep_iteration.append(curr_iteration)
        self.rep_best_cost.append(curr_best_cost)
        self.rep_best_centroids.append(curr_best_centroids)
        