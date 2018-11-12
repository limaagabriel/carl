import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.clustering.evaluation.Metrics import Metrics

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():

    ROOT = '/home/elliackin/Documents/Swarm-Intelligence-Research/Simulation-2007-LA-CCI/2007_LA-CCI_ClusteringSimulation-ID-26-Mai-2017-20h:42m:02s/gap/'
    data = pd.read_excel(
        '/home/elliackin/Documents/Swarm-Intelligence-Research/SRC-Swarm-Intelligence/clustering-optimization/data/Saidafinal4periodo.xlsx',
        sheetname='Plan1')
    data.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)
    x = data.iloc[:, :].values.astype(float)

    x = MinMaxScaler().fit_transform(x)


    (nr,nc) = np.shape(x)


    for i in range(nc):
        values_x = np.unique(x[:, 6])
        values_y = np.unique(x[:, i])
        plt.scatter(x[:, 6], x[:, i], c='k', s=10, edgecolors='k')
        plt.title('Fabiana')
        plt.xlabel('Error 7')
        plt.ylabel('Error ' + str(i+1))
        plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    main()
