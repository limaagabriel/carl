import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tradicional.CPSO import CPSO
from tradicional.KMPSO import KPSO
from tradicional.Metrics import Metrics

from clustering.tradicional.PSC import PSC


def main():

    print("Loading dataset")
    df = pd.read_excel(io='data/Saidafinal4periodo.xlsx', sheetname='Plan1')
    df.drop(['ID', 'Nome', 'E-mail'], 1, inplace=True)

    print("Number of objects in the current dataset: " + str(len(df.index)))
    numberObjects = len(df.index)

    x = df.iloc[0:numberObjects, 0:numberObjects].values.astype(float)

    print("Nomalizing dataset so that all dimenions are in the same scale")
    std = MinMaxScaler()
    x = std.fit_transform(x)

    print("Creating directory to store all clustering solutions")
    PATHNAME_CLUSTERS_SOL = "clusters_solutions"
    directory = os.path.dirname(PATHNAME_CLUSTERS_SOL)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    NUMBER_RUNS = 5

    rng = range(2, 10)
    k_pso = []
    mean = []
    std = []
    print("Start KPSO\n")
    for k in rng:
        print("\t  Number K = " + str(k) + "\n" )
        for i in range(NUMBER_RUNS):
            print("\t\t Run ====> " + str(i))
            clf = KPSO(n_clusters=k, swarm_size=15, n_iter=100, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
            clf.fit(x)
            k_pso.append(Metrics.gap_statistic(x, clf.centroids))
        mean.append(np.mean(k_pso))
        std.append(np.std(k_pso))
        k_pso = []

    plt.figure(0)
    plt.subplot(131)
    plt.title('KPSO - GAP')
    plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
    plt.xlabel('K')
    plt.ylabel('GAP Statistic')

    plot_gap_avaliation(mean, std,131,'KPSO')
    print("KPSO executions are completed!\n\n")

    print("Start CPSO\n")

    cpso = []
    mean = []
    std = []
    for k in rng:
        print("\t  Number K = " + str(k) + "\n")
        for i in range(NUMBER_RUNS):
            print("\t\t Run ====> " + str(i))
            clf = CPSO(n_clusters=k, swarm_size=15, n_iter=100, w=0.72, lb_w=0.4, c1=1.49, c2=1.49)
            clf.fit(x)
            cpso.append(Metrics.gap_statistic(x, clf.centroids))
        mean.append(np.mean(cpso))
        std.append(np.std(cpso))
        cpso = []

    plt.figure(0)
    plt.subplot(132)
    plt.title('CPSO - GAP')
    plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
    plt.xlabel('K')
    plt.ylabel('GAP Statistic')

    plot_gap_avaliation(mean, std, 132, 'CPSO')
    print("CPSO executions are completed!\n\n")

    print("Start PSC\n")

    psc = []
    mean = []
    std = []
    for k in rng:
        print("\t  Number K = " + str(k) + "\n")
        for i in range(NUMBER_RUNS):
            print("\t\t Run ====> " + str(i))
            clf = PSC(minf=0, maxf=1, swarm_size=k, n_iter=200, w=0.95, v_max=0.01)
            clf.fit(x)
            psc.append(Metrics.gap_statistic(x, clf.centroids))
        mean.append(np.mean(psc))
        std.append(np.std(psc))
        psc = []

    plt.figure(0)
    plt.subplot(133)
    plt.title('PSC - GAP')
    plt.errorbar(rng, mean, yerr=std, marker='o', ecolor='b', capthick=2, barsabove=True)
    plt.xlabel('K')
    plt.ylabel('GAP Statistic')

    plot_gap_avaliation(mean, std, 133, 'PSC')
    print("PSC executions are completed!\n")

    plt.tight_layout()
    plt.show()

def plot_gap_avaliation(mean, std, pos_fig, name_alg):
    plt.figure(1)
    plt.subplot(pos_fig)
    plt.title(name_alg + " - GAP")
    mean_menos_std = range(8)
    for i in range(8):
        mean_menos_std[i] = mean[i] - std[i]
    mean_menos_std[0] = mean[0]
    plt.plot(range(2, 10), mean_menos_std, "o-")
    plt.xlabel('K')
    plt.ylabel('GAP Statistic')

if __name__ == '__main__':
    main()