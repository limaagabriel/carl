import os
import numpy as np
import matplotlib.pyplot as plt

min_k = 3
max_k = 15
std_index = 1
mean_index = 0
experiment = 'find_k'
source = 'HuocDataset'
k_range = range(min_k, max_k + 1)


base_path = os.path.join('results', source, experiment, 'FCMeans')
for root, dirs, files in os.walk(base_path):
	if len(files) > 0:
		ks = []
		mean = []
		std_dev = []

		for x in k_range:
			filename = '_{}.csv'.format(x)
			filepath = os.path.join(root, filename)
			data = np.genfromtxt(filepath, delimiter=',')

			ks.append(x)
			mean.append(data[mean_index])
			std_dev.append(data[std_index])

		plt.errorbar(ks, mean, std_dev)	
		plt.savefig(os.path.join(root, 'result.png'))
		plt.clf()
		plt.close()
