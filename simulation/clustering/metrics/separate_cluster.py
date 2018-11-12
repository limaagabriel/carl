__author__ = 'mgmmacedo'

import os
import numpy as np
import pandas as pd

os.chdir('../../../..')

file_name_0 = open("results/all_cities/label.csv", 'rb')
file_name_1 = open("results/all_cities/infomap.csv", 'rb')
file_name_2 = open("results/all_cities/louvain.csv", 'rb')
file_name_3 = open("results/all_cities/stochastic.csv", 'rb')

original = pd.read_csv('data/network_science_project/500_Cities_CDC_organized.csv', header=None, sep=";")
original = pd.DataFrame(original)
original = original.transpose()
ids = []


for i in range(2):
    ids.append(file_name_0.readline().strip().split(','))
    print len(ids[i])

for i in range(0, 2):
    cluster = []
    cluster.append(np.array(original[0]))
    for id_o in range(len(ids[i])):
        for line in range(1, 500):
            if (original[line][0]).strip() == (ids[i][id_o]).strip():
                cluster.append(np.array(original[line]))
    print len(cluster)
    savecluster = pd.DataFrame(cluster)
    savecluster.to_csv("data/network_science_project/cluster_label_{0}.csv".format(str(i)))

file_name_0.close()

print
print
ids = []
for i in range(2):
    ids.append(file_name_1.readline().strip().split(','))
    print len(ids[i])

for i in range(0, 2):
    cluster = []
    cluster.append(np.array(original[0]))
    for id_o in range(len(ids[i])):
        for line in range(1, 500):
            if (original[line][0]).strip() == (ids[i][id_o]).strip():
                cluster.append(np.array(original[line]))
    print len(cluster)
    savecluster = pd.DataFrame(cluster)
    savecluster.to_csv("data/network_science_project/cluster_infomap_{0}.csv".format(str(i)))

file_name_1.close()

print
print
ids = []
for i in range(3):
    ids.append(file_name_2.readline().strip().split(','))
    print len(ids[i])

for i in range(0, 3):
    cluster = []
    cluster.append(np.array(original[0]))
    for id_o in range(len(ids[i])):
        for line in range(1, 500):
            if (original[line][0]).strip() == (ids[i][id_o]).strip():
                cluster.append(np.array(original[line]))
    print len(cluster)
    savecluster = pd.DataFrame(cluster)
    savecluster.to_csv("data/network_science_project/cluster_louvain_{0}.csv".format(str(i)))

file_name_2.close()

print
print
ids = []
for i in range(13):
    ids.append(file_name_3.readline().strip().split(','))
    print len(ids[i])

for i in range(0, 13):
    cluster = []
    cluster.append(np.array(original[0]))
    for id_o in range(len(ids[i])):
        for line in range(1, 500):
            if (original[line][0]).strip() == (ids[i][id_o]).strip():
                cluster.append(np.array(original[line]))
    print len(cluster)
    savecluster = pd.DataFrame(cluster)
    savecluster.to_csv("data/network_science_project/cluster_stochastic_{0}.csv".format(str(i)))

file_name_3.close()

print
print