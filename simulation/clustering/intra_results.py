import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from src.clustering.evaluation.Metrics import Metrics

def convert_df_to_dict(df):
    centroids = {}
    for idx in df.columns:
        centroids[int(idx)] = df[idx].values
    return centroids

def main():

    data = pd.read_excel('//home/elliackin/Documents/Swarm-Intelligence-Research/SRC-Swarm-Intelligence/'
                         'clustering-optimization/data/Saidafinal4periodo.xlsx', sheetname='Plan1')

    data.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)

    x = data.iloc[:, :].values.astype(float)
    print("Nomalizing dataset so that all dimenions are in the same scale")
    min_max_scaler = MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    indices_attributes = np.array([1, 7, 8, 10, 12, 13, 16])
    indices_attributes = indices_attributes - 1
    x = x[:, indices_attributes]

    # FOLDER = KMPSOC | PSC | PSOC -> ALGOS

    algorithms = ['PSOC','KMPSOC', "PSC"]

    mean_metric_by_algorithm = {}
    std_metric_by_algorithm = {}



    for idx_alg in range(len(algorithms)):

        mean_metric_by_algorithm[algorithms[idx_alg]] = []
        std_metric_by_algorithm[algorithms[idx_alg]] = []

        total_mean = []
        total_std  = []

        n_centroids = range(2,10)

        for k in n_centroids:

            # EACH SIMULATION inside n-centroid
            simulation = []

            pathname = "/home/elliackin/Documents/Swarm-Intelligence-Research" \
                           "/Simulation-2007-LA-CCI/2017_LA-CCI_ClusteringSimulation-ID-10-Jun-2017-15h:01m:27s" \
                           "/data/" + algorithms[idx_alg] + "/clusters/" + str(k) +"-centroids/"

            for file in os.listdir(pathname):
                if os.path.join(pathname, file).endswith('.csv'):

                    pathname_temp = os.path.join(pathname, file)

                    #print pathname_temp

                    df = pd.read_csv(pathname_temp)

                    centroids = convert_df_to_dict(df)


                    simulation.append(Metrics.intra_cluster_statistic(x,centroids))

            total_mean.append(np.array(simulation).mean())
            total_std.append(np.array(simulation).std())

        mean_metric_by_algorithm[algorithms[idx_alg]] = total_mean
        std_metric_by_algorithm[algorithms[idx_alg]]  = total_std

    min_value_mean = np.inf
    max_value_mean = -np.inf

    min_value_std = np.inf
    max_value_std = -np.inf

    for idx_alg in range(len(algorithms)):
        curr_min = np.amin(mean_metric_by_algorithm[algorithms[idx_alg]])
        curr_max = np.amax(mean_metric_by_algorithm[algorithms[idx_alg]])
        min_value_mean = np.minimum(min_value_mean, curr_min)
        max_value_mean = np.maximum(max_value_mean, curr_max)

        curr_min = np.amin(std_metric_by_algorithm[algorithms[idx_alg]])
        curr_max = np.amax(std_metric_by_algorithm[algorithms[idx_alg]])
        min_value_std = np.minimum(min_value_std, curr_min)
        max_value_std = np.maximum(max_value_std, curr_max)


    #plt.figure(figsize=(12, 4))

    pathname_out = "/home/elliackin/Documents/Swarm-Intelligence-Research/Simulation-2007-LA-CCI/Figuras LACCI/"

    for idx_alg in range(len(algorithms)):
        total_mean = mean_metric_by_algorithm[algorithms[idx_alg]]
        total_std  = std_metric_by_algorithm[algorithms[idx_alg]]

        plt.figure(idx_alg)
        plt.errorbar(n_centroids, total_mean, yerr=total_std, linewidth=0.5, elinewidth=0.5, color='b')
        plt.plot(n_centroids, total_mean, color='b', marker='o', linewidth=0.5, markersize=5)
        plt.xticks(n_centroids)
        plt.title(algorithms[idx_alg] + ' - INTRA CLUSTER SUM')
        plt.ylabel('Intra Cluster Sum')
        plt.xlabel('Number of Clusters (k)')

        ymin = min_value_mean - min_value_std
        ymax = max_value_mean + max_value_std
        delta = ymax - ymin
        plt.ylim([ymin - 0.5 * delta, ymax + 0.5 * delta])

        plt.tight_layout()
        plt.savefig(pathname_out + algorithms[idx_alg]+ "-SSW.pdf")




if __name__ == '__main__':
    main()