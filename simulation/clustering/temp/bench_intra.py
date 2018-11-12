import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cluster.CPSO import CPSO
from cluster.KMPSO import KPSO
from cluster.PSC import PSC
from sklearn.preprocessing import MinMaxScaler

from cluster.evaluation.Metrics import Metrics


def main():
    df = pd.read_excel(io='data/Saidafinal4periodo.xlsx', sheetname='Plan1')
    df.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)

    x = df.iloc[:, :].values.astype(float)
    std = MinMaxScaler()
    x = std.fit_transform(x)

    df = []
    name = ['KPSO', 'CPSO', 'PSC']
    for i in range(len(name)):
        metrics = []
        mean = []
        std = []
        rng = range(2, 11)
        for k in rng:
            for j in range(30):
                if (name[i] == 'KPSO'):
                    clf = KPSO(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
                elif (name[i] == 'CPSO'):
                    clf = CPSO(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
                elif (name[i] == 'PSC'):
                    clf = PSC(minf=0, maxf=1, swarm_size=k, n_iter=500, w=0.95, v_max=0.01)
                clf.fit(x)
                centroids = pd.DataFrame(clf.centroids)
                centroids.to_csv('intra/' + name[i] + '/' + str(k) + '-centroids/centroid-simulation' + str(j) + '.csv', index=False)
                metrics.append(Metrics.intra_cluster_statistic(data=x, centroids=clf.centroids))
            mean.append(np.mean(metrics))
            std.append(np.std(metrics))
            df.append([name[i], k, np.mean(metrics), np.std(metrics)])
            metrics = []

        plt.subplot(130 + (i + 1))
        plt.title(str(name[i]) + ' - INTRA')
        plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
        plt.xlabel('Clusters')
        plt.ylabel('INTRA Statistic')

    df = pd.DataFrame(df)
    df.columns = ['ALGORITHM', 'CLUSTERS', 'GAP MEAN', 'GAP STD']
    df.to_csv('intra.csv', index=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
