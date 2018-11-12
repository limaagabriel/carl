import pandas as pd
#import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


from src.clustering.evaluation.Metrics import Metrics

def main():


    pathname_dir_results = '/home/elliackin/Documents/Swarm-Intelligence-Research/' \
                           'Simulation-2007-LA-CCI/2007_LA-CCI_ClusteringSimulation-ID-26-Mai-2017-20h:42m:02s'


    pathname_dataset  =  "/home/elliackin/Documents/Swarm-Intelligence-Research/" \
                         "SRC-Swarm-Intelligence/clustering-optimization/data/Saidafinal4periodo.xlsx"


    df = pd.read_csv(pathname_dir_results + '/gap/gap.csv')
    psoc = df[df['ALGORITHM'] == 'PSOC']
    kmpsoc = df[df['ALGORITHM'] == 'KMPSOC']
    psc = df[df['ALGORITHM'] == 'PSC']

    # i = 1
    # algos = [psoc, kmpsoc, psc]
    # plt.figure(figsize=(12,4))
    # for algo in algos:
    #     plt.subplot(130 + i)
    #     plt.errorbar(algo['CLUSTERS'], algo['GAP MEAN'], yerr=algo['GAP STD'], linewidth=0.5, elinewidth=0.5, color='b')
    #     plt.plot(algo['CLUSTERS'], algo['GAP MEAN'], color='b', marker='o', linewidth=0.5, markersize=5)
    #     plt.xticks(algo['CLUSTERS'])
    #     plt.title(algo['ALGORITHM'].values[0] + ' - GAP')
    #     plt.ylabel('GAP Measure')
    #     plt.xlabel('Number of Clusters')
    #     i += 1
    #plt.tight_layout()
    #plt.show()


    print("Loading dataset")

    df = pd.read_excel(io=pathname_dataset, sheetname='Plan1')
    df.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)

    x = df.iloc[:, :].values.astype(float)
    print("Nomalizing dataset so that all dimenions are in the same scale")
    std = MinMaxScaler()
    data = std.fit_transform(x)


    qe = []

    plt.figure(figsize=(12, 4))
    mean = []
    std = []
    for k in range(2,10):
        metrics = []
        for j in range(30):
            curr_directory = pathname_dir_results + '/gap/KMPSOC/' + str(k) + '-centroids'
            filename = curr_directory + '/centroid-simulation-' + str(j) + '.csv'
            df = pd.read_csv(filename)
            cluster =  df.as_matrix()
            raw_centroids = cluster.transpose()
            centroids = {}
            for i in range(k):
                centroids[i] = raw_centroids[i]
            qe_value = Metrics.intra_cluster_statistic(data, centroids)
            metrics.append(qe_value)
        mean.append(np.mean(metrics))
        std.append(np.std(metrics))
    plt.errorbar(range(2,10), mean, yerr=std, linewidth=0.5, elinewidth=0.5, color='b')
    plt.plot(range(2,10), mean, color='b', marker='o', linewidth=0.5, markersize=5)
    plt.xticks(range(2,10))
    plt.title('KMPSOC')
    plt.ylabel('QE Measure')
    plt.xlabel('K')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()