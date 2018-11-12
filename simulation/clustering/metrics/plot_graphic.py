__author__ = 'mgmmacedo'
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.clustering.tradicional.KMeans import KMeans
from src.clustering.tradicional.FCMeans import FCMeans
from src.clustering.evaluation.Metrics import Metrics
from src.clustering.swarm.pso.PSOC import PSOC
from src.clustering.swarm.pso.KMPSOC import KMPSOC
from src.clustering.swarm.pso.PSC import PSC


def main():
    print("Loading dataset")
    rng = range(2, 10)
    # name = "gap"
    # subtitle = 'Gap Statistic (Gap)'
    # name = "quantization_error"
    # subtitle = 'Quantization error (QE)'
    # name = "sum_inter_cluster_dist"
    # subtitle = 'Inter cluster distance (SSB)'
    # name = "sum_intra_cluster_dist"
    # subtitle = 'Intra cluster distance (SSW)'
    # algorithm = "PSOC"
    # algorithm = "KMPSOC"
    # algorithm = "PSOKM"
    # algorithm = "PSC"
    # algorithm = "K-means"

    algorithms = ["PSOC","KMPSOC","PSOKM","PSC","K-means"]
    subtitles = ['Gap Statistic (Gap)','Quantization error (QE)','Inter cluster distance (SSB)','Intra cluster distance (SSW)']
    names = ["gap", "quantization_error", "sum_inter_cluster_dist", "sum_intra_cluster_dist"]

    for alg in range(len(algorithms)):
        for sub in range(len(subtitles)):
            mean = []
            std = []
            for i in rng:
                path = "/Users/mgmmacedo/Documents/1_LACCI/data/"+ algorithms[alg] + "/metrics/{0}".format(str(i))
                dfs = pd.read_csv(path+"-centroids/"+names[sub]+".csv", header=None, sep=' ')
                dfs = dfs.drop(0, 1)
                dfs = dfs.drop(0, 0)
                x = dfs.iloc[:, :].values.astype(float)
                dfs = dfs.iloc[:, :].values.astype(float)
                mean.append(np.mean(dfs))
                std.append(np.std(dfs))
            plt.clf()
            plt.title(str("")+''+algorithms[alg])
            plt.errorbar(rng, mean, yerr=std, marker='s', ecolor='0', color='gray', capsize=5, markersize=10, alpha=0.5, lw=1, capthick=2, barsabove=True)
            plt.margins(0.05, 0.05)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel(subtitles[sub])
            plt.tight_layout()
            saveName = "results/"
            saveName = saveName + "metric_"+names[sub]
            saveName = saveName + "_algorithm_"+algorithms[alg]
            plt.savefig(saveName+"_5_period.pdf")
            all_ms = []
            for i in range(len(mean)):
                all_ms.append([i+2, mean[i],std[i], mean[i]-std[i]])
            saveCSV = pd.DataFrame(all_ms)
            saveCSV.to_csv(saveName+"_5_period.csv")
            # plt.show()


if __name__ == '__main__':
    main()


