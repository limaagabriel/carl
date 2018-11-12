import numpy as np
import pandas as pd
from numpy import genfromtxt

def main(parameters_simulation = None):
    data = genfromtxt('/home/elliackin/Documents/Swarm-Intelligence-Research/Simulation-2007-LA-CCI/2007_LA-CCI_ClusteringSimulation-ID-26-Mai-2017-20h:42m:02s/gap/gap.csv', delimiter=',')

    algorithm = np.zeros((3,8))

    line = 1
    for i  in range(3):
        for j in range(8):
                algorithm[i][j] = (data[line][2]-data[line][3])
                print str(data[line][2]) + " " + str(data[line][3])
                line = line + 1




if __name__ == '__main__':
    main()