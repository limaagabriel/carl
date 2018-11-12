__author__ = 'mgmmacedo'


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir('../../../..')
cluster_0 = pd.read_csv('/Users/mgmmacedo/Documents/SwarmClustering/clustering-optimization/results/CDR_CLODOMIR/results_community_detection/cluster_community_detection/cluster_label_0.csv', sep=";")
cluster_1 = pd.read_csv('/Users/mgmmacedo/Documents/SwarmClustering/clustering-optimization/results/CDR_CLODOMIR/results_community_detection/cluster_community_detection/cluster_label_1.csv', sep=";")

cluster_0

sns.set_style('darkgrid')
sns.pairplot(cluster_0)


cluster_0_corr = cluster_0.corr()
plt.figure(figsize=(12,7))
sns.heatmap(cluster_0_corr, annot=True, cbar=False, cmap='Blues')