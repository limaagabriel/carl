import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from clustering.swarm_cluster.pso.CPSO import CPSO
# from clustering.swarm_cluster.pso.KPSO import KPSO
# from clustering.swarm_cluster.pso.PSC import PSC
from sklearn.preprocessing import MinMaxScaler
from src.clustering.evaluation.Metrics import Metrics
from src.clustering.tradicional.KMeans import KMeans
from src.clustering.tradicional.FCMeans import FCMeans
from sklearn.preprocessing import Normalizer
import csv


def main():
    print("Loading dataset")
    # df = pd.read_excel(io='data/Saidafinal4periodo.xlsx', sheetname='Plan1')
    df = pd.read_csv('data/egyptian-skulls.csv', header=None)
    df = df.drop(len(df.columns)-1, 1)
    x = df[df.apply(lambda x: sum([x_=='?' for x_ in x])==0, axis=1)]
    x = x.iloc[:, :].values.astype(float)
    print("Nomalizing dataset so that all dimensions are in the same scale")
    std = MinMaxScaler()
    x = std.fit_transform(x)
    print x
    SIMULATIONS = 2

    df = []
    name = ['KMeans', 'FCMeans']
    # name = ['KPSO', 'CPSO', 'PSC']
    for i in range(len(name)):
        metrics = []
        mean = []
        std = []
        rng = range(2, 4)
        for k in rng:
            print("Number of Clusters = " + str(k) + "\n")
            for j in range(SIMULATIONS):
                print("Run ====> " + str(j))
                # if (name[i] == 'KPSO'):
                # clf = KPSO(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
                # elif (name[i] == 'CPSO'):
                #     clf = CPSO(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
                # elif (name[i] == 'PSC'):
                #     clf = PSC(minf=0, maxf=1, swarm_size=k, n_iter=500, w=0.95, v_max=0.01)
                if (name[i] == 'KMeans'):
                    clf = KMeans(n_clusters=k)
                elif (name[i] == 'FCMeans'):
                    clf = FCMeans(n_clusters=k)
                clf.fit(x)
                centroids = pd.DataFrame(clf.centroids)
                # if not os.path.exists('gap/' + name[i] + '/' + str(k) + '-centroids'):
                #     os.makedirs('gap/' + name[i] + '/' + str(k) + '-centroids')
                # centroids.to_csv('gap/' + name[i] + '/' + str(k) + '-centroids/centroid-simulation' + str(j) + '.csv', index=False)
                # metrics.append(Metrics.Metrics.inter_cluster_statistic(clf))
                # metrics.append(Metrics.Metrics.cluster_separation_crisp(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.cluster_separation_fuzzy(data=x, clf=clf, m=2.0))
                # metrics.append(Metrics.Metrics.abgss(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.edge_index(data=x, clf=clf, number_neighbors=4))

                # metrics.append(Metrics.Metrics.cluster_connectedness(data=x, clf=clf, number_neighbors=4))

                # metrics.append(Metrics.Metrics.intra_cluster_statistic(clf))
                # metrics.append(Metrics.Metrics.ball_hall(data=x, clf=clf))
                # metrics.append( Metrics.Metrics.j_index(data=x, clf=clf, m=2.0) )
                # metrics.append( Metrics.Metrics.total_within_cluster_variance(data=x, clf=clf) )
                # metrics.append(Metrics.Metrics.classification_entropy(data=x, clf=clf, m=2.0))

                # metrics.append(Metrics.Metrics.intra_cluster_entropy(data=x, clf=clf))

                # metrics.append(Metrics.Metrics.variance_based_ch(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.hartigan(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.xu(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.rl(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.wb(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.xie_beni(data=x, clf=clf, m=2.0))
                # c = clf.centroids[0][0]
                # metrics.append(Metrics.Metrics.i_index(data=x, clf=clf, centroidUnique=c))
                # metrics.append(Metrics.Metrics.dunn_index(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.davies_bouldin(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.cs_index(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.silhouette(data=x, clf=clf))
                # metrics.append(Metrics.Metrics.min_max_cut(data=x, clf=clf))
                metrics.append(Metrics.Metrics.gap_statistic(data=x, clf=clf))
            mean.append(np.mean(metrics))
            std.append(np.std(metrics))
            df.append([name[i], k, np.mean(metrics), np.std(metrics)])
            metrics = []

        plt.subplot(130 + (i + 1))
        plt.title(str(name[i]) + ' - Metric')
        plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
        plt.xlabel('Clusters')
        plt.ylabel('Metric')
    # df = pd.DataFrame(df)
    # df.columns = ['ALGORITHM', 'CLUSTERS', 'GAP MEAN' , 'GAP STD']
    # df.to_csv('gap.csv', index=False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
