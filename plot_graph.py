from experiments.researcher import Researcher
from clustering.traditional.FCMeans import FCMeans
from data.huoc.datasets import *
import numpy as np
import itertools
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

min_k = 3
max_k = 20
sample_size = 30
sample_range = range(sample_size)
sources = [HuocDataset, CognitiveTestsDataset, SocialInfoDataset, SocialAndTestsDataset, GeneticMarkersDataset]
k_range = range(min_k, max_k + 1)
algorithms = [FCMeans]

total = len(algorithms) * len(k_range)
labels = list(map(lambda x: x.name,sources))
for source_index, source in enumerate(sources):
	experiment = Researcher(source())
	accuracy = np.zeros((sample_size, len(k_range)))

	for algorithm, k in tqdm(itertools.product(algorithms, k_range), total=total):
		for iteration in sample_range:
			_, right, wrong =  experiment.confusion_labels_kernel(algorithm, k)

			a = float(len(right))
			b = float(len(wrong))
			accuracy[iteration, k - min_k] = a / (a + b)

	mean = accuracy.mean(axis=0)
	std = accuracy.std(axis=0)

	print(mean, std)
	plt.errorbar(k_range, mean, std)

plt.xlabel('Number of clusters')
plt.ylabel('Accuracy')
plt.legend(labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)

plt.savefig('accuracy.png', bbox_inches='tight')
plt.clf()
plt.close()
		