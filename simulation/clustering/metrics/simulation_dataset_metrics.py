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
from sklearn.utils import shuffle
import os

def main():
    print("Loading dataset")
    os.chdir('../../../..')
    dfs = [pd.read_csv('data/empty.csv', header=None) for k in range(28)]
    # iris pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    dfs[0] = pd.read_csv('data/datasets_metrics/iris.csv', header=None)
    # wine pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    dfs[1] = pd.read_csv('data/datasets_metrics/wine.csv', header=None)
    # glass pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', header=None)
    dfs[2] = pd.read_csv('data/datasets_metrics/glass.csv', header=None)
    # breast cancer wincosin pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
    dfs[3] = pd.read_csv('data/datasets_metrics/breast-cancer-wisconsin.csv', header=None)
    # wdbc pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    dfs[4] = pd.read_csv('data/datasets_metrics/wdbc.csv', header=None)
    # liver disorders pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data', header=None)
    dfs[5] = pd.read_csv('data/datasets_metrics/bupa.csv', header=None)
    # contraceptive method choice pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data', header=None)
    dfs[6] = pd.read_csv('data/datasets_metrics/cmc.csv', header=None)
    # tiroide pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data', header=None)
    dfs[7] = pd.read_csv('data/datasets_metrics/new-thyroid.csv', header=None)
    # dematology pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data', header=None)
    dfs[8] = pd.read_csv('data/datasets_metrics/dermatology.csv', header=None)
    # egyptian skools http://www.dm.unibo.it/~simoncin/EgyptianSkulls.html
    dfs[9] = df = pd.read_csv('data/datasets_metrics/egyptian-skulls.csv', header=None)
    # heart statlog pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat', header=None)
    dfs[10] = df = pd.read_csv('data/datasets_metrics/heart.csv', header=None)
    # ionosphere
    dfs[11] = df = pd.read_csv('data/datasets_metrics/ionosphere.csv', header=None)
    # vehicle
    dfs[12] = df = pd.read_csv('data/datasets_metrics/vehicle.csv', header=None)
    # balance scale pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', header=None)
    dfs[13] = df = pd.read_csv('data/datasets_metrics/balance-scale.csv', header=None)
    # sonar
    dfs[14] = df = pd.read_csv('data/datasets_metrics/sonar.csv', header=None)
    # zoo pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', header=None)
    dfs[15] = df = pd.read_csv('data/datasets_metrics/zoo.csv', header=None)
    # isolet5
    dfs[16] = df = pd.read_csv('data/datasets_metrics/isolet5.csv', header=None)
    # movement libras
    dfs[17] = df = pd.read_csv('data/datasets_metrics/libras.csv', header=None)
    # cleveland http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
    dfs[18] = df = pd.read_csv('data/datasets_metrics/cleveland.csv', header=None)
    # australian
    dfs[19] = df = pd.read_csv('data/datasets_metrics/australian.csv', header=None)
    dfs[20] = df = pd.read_csv('data/shapesets/compound.csv', header=None)
    dfs[21] = df = pd.read_csv('data/shapesets/flame.csv', header=None)
    dfs[22] = df = pd.read_csv('data/shapesets/jain.csv', header=None)
    dfs[23] = df = pd.read_csv('data/shapesets/r15.csv', header=None)
    dfs[24] = df = pd.read_csv('data/shapesets/d31.csv', header=None)
    dfs[25] = df = pd.read_csv('data/shapesets/spiral.csv', header=None)
    dfs[26] = df = pd.read_csv('data/shapesets/pathbased.csv', header=None)
    dfs[27] = df = pd.read_csv('data/shapesets/agregation.csv', header=None)
    # hill-valley
    # diabetes
    # olive
    # crud oil
    # musk version 1
    # landsat satellite
    # heart disease len(dfs)
    for eachdataset in range(0, len(dfs)):
        df = dfs[eachdataset]
        df = df.drop(len(df.columns) - 1, 1)
        x = df[df.apply(lambda x: sum([x_ == '?' for x_ in x]) == 0, axis=1)]
        x = x.iloc[:, :].values.astype(float)
        print("Nomalizing dataset so that all dimenions are in the same scale")
        std = MinMaxScaler()
        x = std.fit_transform(x)
        x = x[np.random.permutation(len(x))]
        SIMULATIONS = 30
        df = []
        # name = ['KMeans', 'FCMeans']
        name = ['FCMeans']
        # name = ['KPSO', 'CPSO', 'PSC']
        for i in range(len(name)):
            metrics = []
            mean = []
            std = []
            rng = range(2, 10)
            # 27
            for metricNumber in range(0, 26):
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
                            clf = FCMeans(n_clusters=k, n_iter=1000)
                        elif (name[i] == 'PSOC'):
                            clf = PSOC(n_clusters=k, swarm_size=15, n_iter=500, w=0.72, c1=1.49, c2=1.49)
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
                        clusters = {}
                        for c in clf.centroids:
                            clusters[c] = []

                        for xi in x:
                            dist = [np.linalg.norm(xi - clf.centroids[c]) for c in clf.centroids]
                            class_ = dist.index(min(dist))
                            clusters[class_].append(xi)

                        # precisa inverter?
                        file = open(sn+"_clusters.csv", 'w')
                        file.write(str(len(clf.centroids)) + '\n')
                        for c in range(len(clf.centroids)):
                            file.write(str(len(clusters[c])) + '\n')
                            for xi in range(len(clusters[c])):
                                file.write(str(clusters[c][xi][0]))
                                for xij in range(1, len(clusters[c][xi])):
                                    file.write(' ' + str(clusters[c][xi][xij]))
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

