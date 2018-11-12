import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.clustering.swarm.pso.PSC import PSC
from src.clustering.swarm.pso.PSOC import PSOC
from sklearn.preprocessing import MinMaxScaler
from src.clustering.swarm.pso.KMPSOC import KMPSOC
from src.clustering.swarm.pso.PSOCKM import PSOCKM
from src.clustering.tradicional.KMeans import KMeans
from src.clustering.evaluation.Metrics import Metrics
from src.clustering.tradicional.FCMeans import FCMeans


def main():
    num_exec = 30
    swarm_size = 30
    num_iter = 1000
    # names = ['KMeans', 'FCMeans', 'PSOC', 'PSC', 'KMPSOC', 'PSOCKM']
    names = ['PSC', 'PSOC', 'KMPSOC', 'PSOCKM']
    # names = ['PSOC', 'PSC', 'KMPSOC', 'PSOCKM']

    print("Loading dataset")
    os.chdir('../../..')
    df = pd.read_csv('data/booking_website/booking_website_without_empty_values.csv')
    df = df.drop(['id'], axis=1)
    # df = df.drop(['idade'], axis=1)
    df = df.drop(['sexo'], axis=1)
    x = df[df.apply(lambda x: sum([x_ == '?' for x_ in x]) == 0, axis=1)]
    x = x.iloc[:, :].values.astype(float)

    print("Nomalizing dataset so that all dimenions are in the same scale")
    std = MinMaxScaler()
    x = std.fit_transform(x)
    x = x[np.random.permutation(len(x))]
    for i in range(len(names)):
        metrics = []
        rng = range(2, 11)
        for metricNumber in ["intraClusterStatistic", "quantizationError", "sumInterClusterDistance"]:
        # for metricNumber in ["gap"]:
            print("Algorithm: " + names[i])
            mean = []
            std = []
            dff = []
            for k in rng:
                # print(" Number of Clusters = " + str(k))
                for j in tqdm(range(num_exec)):
                    if names[i] == 'KPSO':
                        clf = KMPSOC(n_clusters=k, swarm_size=swarm_size, n_iter=num_iter, w=0.72, c1=1.49, c2=1.49)
                    elif names[i] == 'PSOC':
                        clf = PSOC(n_clusters=k, swarm_size=swarm_size, n_iter=num_iter, w=0.72, c1=1.49, c2=1.49)
                    elif names[i] == 'PSC':
                        clf = PSC(swarm_size=k, n_iter=num_iter, w=0.95, c1=2.05, c2=2.05, c3=1.0, c4=1.0, v_max=0.001)
                    elif names[i] == 'PSOCKM':
                        clf = PSOCKM(n_clusters=k, swarm_size=swarm_size, n_iter=num_iter, w=0.72, c1=1.49, c2=1.49)
                    elif names[i] == 'KMeans':
                        clf = KMeans(n_clusters=k, n_iter=num_iter, shuffle=True, tolerance=0.00001)
                    elif names[i] == 'FCMeans':
                        clf = FCMeans(n_clusters=k, n_iter=num_iter, fuzzy_c=2, tolerance=0.001)

                    clf.fit(x)
                    out_dir = "results/booking/algorithm_{}/metric_{}/".format(names[i], metricNumber)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    file_name = out_dir + "{}_k_{}_exec_{}.csv".format('centroids', k, j)
                    save_centroids = pd.DataFrame(clf.centroids)
                    save_centroids = save_centroids.transpose()
                    save_centroids.to_csv(file_name)

                    clusters = {}
                    for c in clf.centroids:
                        clusters[c] = []

                    for xi in x:
                        dist = [np.linalg.norm(xi - clf.centroids[c]) for c in clf.centroids]
                        class_ = dist.index(min(dist))
                        clusters[class_].append(xi)

                    clusters_file = open(out_dir + "{}_k_{}_exec_{}.csv".format('clusters', k, j), 'w')
                    clusters_file.write(str(len(clf.centroids)) + '\n')

                    for c in range(len(clf.centroids)):
                        clusters_file.write(str(len(clusters[c])) + '\n')
                        for xi in range(len(clusters[c])):
                            clusters_file.write(str(clusters[c][xi][0]))
                            for xij in range(1, len(clusters[c][xi])):
                                clusters_file.write(' ' + str(clusters[c][xi][xij]))
                            clusters_file.write('\n')
                    clusters_file.close()

                    c = clf.centroids[0][0]

                    metrics.append(
                        Metrics.clustering_evaluation("{}".format(metricNumber), centroids=clf.centroids, data=x,
                                                      clf=clf, m=2.0, number_neighbors=2, centroidUnique=c))
                mean.append(np.mean(metrics))
                std.append(np.std(metrics))
                dff.append([names[i], k, np.mean(metrics), np.std(metrics)])
                metrics = []

            # plt.subplot(130 + (i + 1))
            plt.figure()

            figure_name = "results/booking/algorithm_{}/metric_{}/plot.png".format(names[i], metricNumber)
            plt.title(str(names[i]) + ' - Metric ' + metricNumber)
            plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
            plt.xlabel('Clusters')
            plt.ylabel('Metric')
            plt.tight_layout()
            plt.savefig(figure_name)

            save_name = "results/booking/algorithm_{}/metric_{}/output.csv".format(names[i], metricNumber)
            dff = pd.DataFrame(dff)
            dff.columns = ['ALGORITHM', 'CLUSTERS', 'MEAN', 'STD']
            dff.to_csv(save_name)

        # plt.show()


if __name__ == '__main__':
    main()
