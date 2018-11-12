import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from src.clustering.swarm.pso.PSC import PSC
from src.clustering.swarm.pso.PSOC import PSOC
from sklearn.preprocessing import MinMaxScaler
from src.clustering.swarm.pso.KMPSOC import KMPSOC
from src.clustering.swarm.pso.PSOCKM import PSOCKM
from src.clustering.tradicional.KMeans import KMeans
from src.clustering.tradicional.FCMeans import FCMeans


def convert_and_save(clf, centroid_file_name, clusters_file_name, x, std, df_orig, iter):
    centroids = []
    for centroid in clf.centroids.values():
        centroids.append(std.inverse_transform([centroid])[0])
        #centroids.append(centroid)
    save_centroids = pd.DataFrame(centroids)
    save_centroids.to_csv(centroid_file_name)

    clusters = {}
    for c in clf.centroids:
        clusters[c] = []

    for xi in x:
        d = std.inverse_transform([xi])[0]
        d = [str(d[0])[0:8], str(d[1])[0:8]]
        city = df_orig[(df_orig['latitude'].str.contains(d[0])) & (df_orig['longitude'].str.contains(d[1]))]
        # print(d)
        # print(city)
        city = np.asarray(city)[0]

        dist = [np.linalg.norm(xi - clf.centroids[c]) for c in clf.centroids]
        class_ = dist.index(min(dist))
        clusters[class_].append(city)

    clusters_file = open(clusters_file_name, 'w+')

    for c in range(len(clf.centroids)):
        clusters_file.write(str(iter) + '\n')
        for xi in range(len(clusters[c])):
            clusters_file.write(str(clusters[c][xi][0]))
            for xij in range(1, len(clusters[c][xi])):
                clusters_file.write(', ' + str(clusters[c][xi][xij]))
            clusters_file.write('\n')
    clusters_file.close()


def get_data_from_gml(gml_file):
    content = nx.read_gml(gml_file, label='id')
    content = content.nodes._nodes
    col_names = ['id', 'gini', 'country', 'pib', 'longitude', 'city', 'state', 'idh', 'latitude', 'type', 'population']
    res = []
    for key, value in content.iteritems():
        line = [key]
        for key2, value2 in value.iteritems():
            if 'PE' in str(value2):
                line += (str(value2).split(','))
            else:
                line.append(str(value2))
        res.append(line)
    res = pd.DataFrame(res)
    res.columns = col_names
    res = res[['country', 'state', 'city', 'latitude', 'longitude', 'gini', 'pib', 'idh', 'population', 'id']]
    return res


def main():
    num_exec = 30
    swarm_size = 30
    num_iter = 5000
    data_file = 'data/cities_clustering/cities_information.gml'
    algorithms = ['KMPSOC', 'PSOCKM', 'KMeans', 'FCMeans', 'PSOC', 'PSC']

    print("Loading dataset and preprocessing")
    os.chdir('../../..')
    df_orig = get_data_from_gml(data_file)

    # Removing unused attributes
    df = df_orig.drop(['country', 'state', 'city', 'gini', 'pib', 'idh', 'population', 'id'], axis=1)

    # Converting data structure
    x = df[df.apply(lambda x: sum([x_ == '?' for x_ in x]) == 0, axis=1)]
    x = x.iloc[:, :].values.astype(float)

    # Normalizing data set so that all dimensions are in the same scale
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    x = x[np.random.permutation(len(x))]
    for i in range(len(algorithms)):
        rng = range(4, 31)
        print("Algorithm: " + algorithms[i])
        for k in tqdm(rng):
            for j in range(num_exec):
                if algorithms[i] == 'KMPSOC':
                    clf = KMPSOC(n_clusters=k, swarm_size=swarm_size, n_iter=num_iter, w=0.72, c1=1.49, c2=1.49)
                elif algorithms[i] == 'PSOC':
                    clf = PSOC(n_clusters=k, swarm_size=swarm_size, n_iter=num_iter, w=0.72, c1=1.49, c2=1.49)
                elif algorithms[i] == 'PSC':
                    clf = PSC(swarm_size=k, n_iter=num_iter, w=0.95, c1=2.05, c2=2.05, c3=1.0, c4=1.0, v_max=0.001)
                elif algorithms[i] == 'PSOCKM':
                    clf = PSOCKM(n_clusters=k, swarm_size=swarm_size, n_iter=num_iter, w=0.72, c1=1.49, c2=1.49)
                elif algorithms[i] == 'KMeans':
                    clf = KMeans(n_clusters=k, n_iter=num_iter, shuffle=True, tolerance=0.00001)
                elif algorithms[i] == 'FCMeans':
                    clf = FCMeans(n_clusters=k, n_iter=num_iter, fuzzy_c=2, tolerance=0.001)

                iter = clf.fit(x)
                out_dir = "results/cities_clustering/algorithm_{}/".format(algorithms[i])
                # print(iter)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                clusters_file = out_dir + "{}_k_{}_exec_{}.csv".format('clusters', k, j, iter)
                centroids_file = out_dir + "{}_k_{}_exec_{}.csv".format('centroids', k, j, iter)

                # convert_and_save(clf, centroids_file, clusters_file, x, scaler, df_orig)
                convert_and_save(clf, centroids_file, clusters_file, x, scaler, df_orig, iter)


if __name__ == '__main__':
    main()
