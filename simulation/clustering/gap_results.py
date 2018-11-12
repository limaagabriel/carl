import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv('/home/elliackin/Documents/'
                     'Swarm-Intelligence-Research/Simulation-2007-LA-CCI/'
                     '2017_LA-CCI_ClusteringSimulation-ID-10-Jun-2017-15h:01m:27s/data/gap_metric.csv')

    psoc = df[df['ALGORITHM'] == 'PSOC']
    kmpsoc = df[df['ALGORITHM'] == 'KMPSOC']
    psc = df[df['ALGORITHM'] == 'PSC']

    algos = [psoc, kmpsoc, psc]

    mean_min_y = np.inf
    std_min_y  = np.inf
    mean_max_y = -np.inf
    std_max_y  = -np.inf

    for algo in algos:
        mean = algo['GAP MEAN']
        std  = algo['GAP STD']

        mean_min_y = np.minimum(np.amin(mean),mean_min_y)
        std_min_y  = np.minimum(np.amin(std),std_min_y)
        mean_max_y = np.maximum(np.amax(mean),mean_max_y)
        std_max_y = np.maximum(np.amax(std),std_max_y)

    pathname_out = "/home/elliackin/Documents/Swarm-Intelligence-Research/Simulation-2007-LA-CCI/Figuras LACCI/"

    i = 1
    #plt.figure(figsize=(12,4))
    for algo in algos:
        #plt.subplot(130 + i)
        plt.figure(i)
        plt.errorbar(algo['CLUSTERS'], algo['GAP MEAN'], yerr=algo['GAP STD'], linewidth=0.5, elinewidth=0.5, color='b')
        plt.plot(algo['CLUSTERS'], algo['GAP MEAN'], color='b', marker='o', linewidth=0.5, markersize=5)
        plt.xticks(algo['CLUSTERS'])
        plt.title(algo['ALGORITHM'].values[0] + ' - GAP')
        plt.ylabel('GAP Measure')
        plt.xlabel('Number of Clusters (K)')

        ymin = mean_min_y - std_min_y
        ymax = mean_max_y + std_max_y
        delta = ymax - ymin

        plt.ylim([ymin-0.1*delta,ymax+0.1*delta])
        plt.tight_layout()
        plt.savefig(pathname_out + algo['ALGORITHM'].values[0] + "-GAP.pdf")
        i = i + 1

if __name__ == '__main__':
    main()