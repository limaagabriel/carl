from experiments.researcher import Researcher
from clustering.traditional.FCMeans import FCMeans
from data.huoc.datasets import *
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

min_k = 3
max_k = 30
sample_size = 30
sample_range = range(sample_size)
sources = [HuocDataset, CognitiveTestsDataset, SocialInfoDataset, SocialAndTestsDataset, GeneticMarkersDataset]
k_range = range(min_k, max_k + 1)
algorithms = [FCMeans]

class_names = {
	'1': 'Dementia',
	'10': 'MCI',
	'100': 'Control'
}

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

	plt.errorbar(k_range, accuracy.mean(axis=0), accuracy.std(axis=0))

plt.xlabel('Number of clusters')
plt.ylabel('Accuracy')
plt.legend(labels)

plt.savefig('accuracy.png')
plt.clf()
plt.close()
		