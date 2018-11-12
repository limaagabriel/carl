import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clustering.swarm.pso.PSC import PSC


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[:, 4].values
    y[y == "Iris-setosa"] = 0
    y[y == "Iris-versicolor"] = 1
    y[y == "Iris-virginica"] = 2
    x = df.iloc[:, [0, 2]].values

    x = np.array(x)
    y = np.array(y)

    x[:, 0] = (x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())
    x[:, 1] = (x[:, 1] - x[:, 1].min()) / (x[:, 1].max() - x[:, 1].min())

    clf = PSC(swarm_size=3, n_iter=200, w=0.95, v_max=0.01)
    clf.fit(x)


    #Here


    labels = clf.predict(x)

    symbols = ['<', 's', 'o']
    iris_symbols = []
    for i in range(len(y)):
        iris_symbols.append(symbols[y[i]])

    colors = ['g', 'k', 'b']

    clusters_colors = []
    for i in range(len(labels)):
        j = labels[i] % len(colors)
        clusters_colors.append(colors[j])



    for i in range(len(clusters_colors)):
        plt.scatter(x[i, 0], x[i, 1], marker=iris_symbols[i], c=clusters_colors[i], s=50, edgecolors='k')

    for k in clf.centroids:
        plt.scatter(clf.centroids[k][0], clf.centroids[k][1], marker='x', s=100, c='r', linewidth='2')


    #After



    plt.title('PSC')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.show()

if __name__ == '__main__':
    main()
