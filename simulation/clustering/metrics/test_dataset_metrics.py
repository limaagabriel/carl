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
    eachdataset = 072017
    df = pd.read_excel(io='data/Saidafinal4periodo.xlsx', sheetname='Plan1')
    df.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)

    x = df.iloc[:, :].values.astype(float)
    print("Nomalizing dataset so that all dimenions are in the same scale")
    std_metric_by_algorithm_and_k = MinMaxScaler()
    x = std_metric_by_algorithm_and_k.fit_transform(x)



    indices_attributes = np.array([1,7,8,10,12,13,16])
    indices_attributes = indices_attributes -1
    x = x[:,indices_attributes]
    SIMULATIONS = 30

    # name = ['KMeans', 'FCMeans']
    name = ['PSOC']
    # name = ['KPSO', 'CPSO', 'PSC']
    for i in range(len(name)):
        metrics = []
        mean = []
        std = []
        rng = range(2, 10)
        for metricNumber in range(1, 26):
            for k in rng:
                print("Number of Clusters = " + str(k) + "\n")
                for j in range(SIMULATIONS):
                    print("Run ====> " + str(j))
                    # if (name[i] == 'KPSO'):
                    # clf = KPSO(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
                    # elif (name[i] == 'CPSO'):
                    # clf = CPSO(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
                    # elif (name[i] == 'PSC'):
                    #     clf = PSC(minf=0, maxf=1, swarm_size=k, n_iter=500, w=0.95, v_max=0.01)
                    if (name[i] == 'KMeans'):
                        clf = KMeans(n_clusters=k)
                    elif (name[i] == 'FCMeans'):
                        clf = FCMeans(n_clusters=k)
                    elif (name[i] == 'PSOC'):
                        clf = PSOC(n_clusters=k, swarm_size=30, n_iter=1000, w=0.72, c1=1.49, c2=1.49)
                    clf.fit(x)
                    if not os.path.isdir("results/metrics/"+"dataset_{0}".format(str(eachdataset))+"/metric_{0}".format(str(metricNumber))+"/algorithm_{0}".format(str(name[i]))+"/"):
                        os.makedirs("results/metrics/"+"dataset_{0}".format(str(eachdataset))+"/metric_{0}".format(str(metricNumber))+"/algorithm_{0}".format(str(name[i]))+"/")
                    sn = "results/metrics/"+"dataset_{0}".format(str(eachdataset))+"/metric_{0}".format(str(metricNumber))+"/algorithm_{0}".format(str(name[i])+"/")
                    sn = sn + "dataset_{0}".format(str(eachdataset))
                    sn = sn + "_metric_{0}".format(str(metricNumber))
                    sn = sn + "_algorithm_{0}".format(str(name[i]))
                    sn = sn + "_k_{0}".format(str(k))
                    sn = sn + "_simulation_{0}".format(str(j))
                    savecentroids = pd.DataFrame(clf.centroids)
                    savecentroids = savecentroids.transpose()
                    savecentroids.to_csv(sn+"_centroids.csv")
                    # clust = Metrics.get_clusters(x, clf.centroids)
                    # # print np.array(clust[0])
                    # c = []
                    # for ii in range(len(clust)):
                    #     c.append(np.array(clust[ii]))
                    # sc = pd.DataFrame(c)
                    # # precisa inverter?
                    # sc.to_csv(sn+"_clusters.csv")

                    file = open(sn+"_clusters.csv", 'w')
                    file.write(str(len(clf.centroids)) + '\n')
                    file.write(str(clf.solution.number_of_effective_clusters)+ '\n')
                    for c in range(len(clf.centroids)):
                        file.write(str(len(clf.solution.clusters[c])) + '\n')
                        for xi in range(len(clf.solution.clusters[c])):
                            file.write(str(clf.solution.clusters[c][xi][0]))
                            for xij in range(1,len(clf.solution.clusters[c][xi])):
                                file.write(' ' + str(clf.solution.clusters[c][xi][xij]))
                            file.write('\n')
                    file.close()
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
                    c = clf.centroids[0][0]
                    # metrics.append(Metrics.Metrics.i_index(data=x, clf=clf, centroidUnique=c))
                    # metrics.append(Metrics.Metrics.dunn_index(data=x, clf=clf))
                    # metrics.append(Metrics.Metrics.davies_bouldin(data=x, clf=clf))
                    # metrics.append(Metrics.Metrics.cs_index(data=x, clf=clf))
                    # metrics.append(Metrics.Metrics.silhouette(data=x, clf=clf))
                    # metrics.append(Metrics.Metrics.min_max_cut(data=x, clf=clf))
                    # metrics.append(Metrics.Metrics.gap_statistic(data=x, clf=clf))
                    metrics.append(Metrics.clustering_evaluation("{0}".format(str(metricNumber)), centroids=clf.centroids, data=x, clf=clf, m=2.0, number_neighbors=2, centroidUnique=c))
                mean.append(np.mean(metrics))
                std.append(np.std(metrics))
                df.append([name[i], k, np.mean(metrics), np.std(metrics)])
                metrics = []

            # plt.subplot(130 + (i + 1))
            plt.clf()
            plt.title(str(name[i]) + ' - Metric')
            plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
            plt.xlabel('Clusters')
            plt.ylabel('Metric')
            saveName = "results/metrics/"+"dataset_{0}".format(str(eachdataset))+"/metric_{0}".format(str(metricNumber))+"/algorithm_{0}".format(str(name[i])+"/")
            saveName = saveName + "dataset_{0}".format(str(eachdataset))
            saveName = saveName + "_metric_{0}".format(str(metricNumber))
            saveName = saveName + "_algorithm_{0}".format(str(name[i]))
            plt.savefig(saveName+".pdf")
            df = pd.DataFrame(df)
            df.columns = ['ALGORITHM', 'CLUSTERS', 'MEAN', 'STD']
            df.to_csv(saveName+".csv")
            mean = []
            std = []
            plt.tight_layout()
        # plt.show()


if __name__ == '__main__':
    main()

