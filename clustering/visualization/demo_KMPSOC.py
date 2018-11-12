import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.clustering.swarm.pso.KMPSOC import KMPSOC


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[:, 4].values
    y[y == "Iris-setosa"] = 0
    y[y == "Iris-versicolor"] = 1
    y[y == "Iris-virginica"] = 2
    x = df.iloc[:, [0, 2]].values

    x = np.array(x)
    y = np.array(y)

    for i in range(2):
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())

    LB_W = 0.72984
    UP_W = 0.72984
    C_1 = 0.72984 * 2.05
    C_2 = 0.72984 * 2.05

    clf = KMPSOC(n_clusters=3, swarm_size=10, n_iter=100, up_w=UP_W, lb_w=LB_W, c1=C_1, c2=C_2, v_max=0.5)
    clf.fit(x)

    labels = clf.predict(x)

    symbols = ['<', 's', 'o']
    iris_symbols = []
    for i in range(len(y)):
        iris_symbols.append(symbols[y[i]])

    colors = ['g', 'k', 'b']

    clusters_colors = []
    for i in range(len(labels)):
        clusters_colors.append(colors[labels[i]])

    plt.subplot(121)

    for i in range(len(clusters_colors)):
        plt.scatter(x[i, 0], x[i, 1], marker=iris_symbols[i], c=clusters_colors[i], s=50, edgecolors='k')

    for k in clf.centroids:
        plt.scatter(clf.centroids[k][0], clf.centroids[k][1], marker='x', s=100, c='r', linewidth='5')

    plt.title('KMPSOC')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')

    plt.subplot(122)
    plt.plot([(i + 1) for i in range(len(clf.pso.optimum_cost_tracking))], clf.pso.optimum_cost_tracking, c='b')
    plt.legend(["Intra Cluster Cost"])
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')

    plt.show()


if __name__ == '__main__':
    main()
