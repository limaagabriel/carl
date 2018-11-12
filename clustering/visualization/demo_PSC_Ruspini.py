import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from src.clustering.swarm.pso.PSC import PSC


def main():
    #df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df = pd.read_csv('/home/elliackin/Documents/Swarm-Intelligence-Research/SRC-Swarm-Intelligence/'
                     'clustering-optimization/data/test/ruspini.csv')



    x = df.iloc[:, [1, 2]].values

    x = np.array(x,dtype=np.float64)


    x[:, 0] = (x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())
    x[:, 1] = (x[:, 1] - x[:, 1].min()) / (x[:, 1].max() - x[:, 1].min())

    clf = PSC(swarm_size=8, n_iter=200, w=0.95, v_max=0.01)
    clf.fit(x)


    #Here





    plt.scatter(x[:, 0], x[:, 1], marker='o', c='b', s=50, edgecolors='k')

    for k in clf.centroids:
        plt.scatter(clf.centroids[k][0], clf.centroids[k][1], marker='x', s=100, c='r', linewidth='2')


    #After



    plt.title('PSC')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.show()

if __name__ == '__main__':
    main()
