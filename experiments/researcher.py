import math
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from clustering.evaluation.Metrics import Metrics
from experiments.helpers.decorators import *


class Researcher(object):
	def __init__(self, source):
		self.source = source
		self.classes = source.classes
		self.full_dataset = source.dataset
		self.dataset = source.unlabeled_data
		
	def __unsupervised_confusion_matrix(self, clusters):
		matrix = {
			str(int(a)) : {
				str(int(b)) : 0 for b in self.classes
			} for a in self.classes
		}

		right = []
		wrong = []

		for cluster in clusters:
			if cluster.shape[0] == 0:
				continue

			classes = cluster[:,-1]
			values, counts = np.unique(classes, return_counts=True)
			max_index = np.argmax(counts)
			most_frequent_class = str(int(values[max_index]))

			for data in cluster:
				current_class = str(int(data[-1]))
				matrix[current_class][most_frequent_class] += 1

				if current_class == most_frequent_class:
					right.append((data, most_frequent_class))
				else:
					wrong.append((data, most_frequent_class))
		
		return matrix, right, wrong

	def __run_pca(self, dataset, n_components):
		scaler = StandardScaler()
		scaled_dataset = scaler.fit_transform(dataset)

		pca = PCA(n_components=n_components)
		return pca.fit_transform(scaled_dataset)

	@prepare_directory('results')
	@scatter_plot
	def visualize_pca(self, dataset, assignments, k, algorithm, n_components):
		if n_components < 2:
			n_components = 2
		if n_components > 3:
			n_components = 3

		plot_id = '{}_{}_{}'.format(algorithm.__name__, k, n_components)
		embedded_dataset = self.__run_pca(dataset, n_components)
		
		return (plot_id, embedded_dataset, assignments)

	@iterate('k_range', 'algorithms')
	def visualize(self, k, algorithm):
		instance = algorithm(n_clusters=k)
		instance.fit(self.dataset)
		assignments = map(lambda x: instance.predict(x), self.dataset)

		self.visualize_pca(self.dataset, assignments, k, algorithm, 2)
		self.visualize_pca(self.dataset, assignments, k, algorithm, 3)

	def confusion_labels_kernel(self, algorithm, k):
		return self.__confusion_labels_kernel(algorithm, k)

	def __confusion_labels_kernel(self, algorithm, k):
		instance = algorithm(n_clusters=k)
		instance.fit(self.dataset)

		clusters = [[] for _ in range(k)]
		for data in self.full_dataset:
			unclassified_data = data[:-1]
			predicted = instance.predict(unclassified_data)

			clusters[predicted].append(data)

		clusters = list(map(lambda x: np.array(x), clusters))
		return self.__unsupervised_confusion_matrix(clusters)

	@prepare_directory('results')
	@iterate('algorithms', 'k_range')
	@confusion_matrix
	def confusion_labels(self, algorithm, k):
		plot_id = '{}_{}'.format(algorithm.__name__, k)
		return plot_id, self.__confusion_labels_kernel(algorithm, k)[0]

	@clean('results')
	@prepare_directory('results')
	@errorbar('k_range', 0, 1)
	@iterate('algorithms', 'k_range')
	@save_results_with_statistical_data
	def find_k(self, algorithm, k, sampling_size, metrics):
		filename = '_{}.csv'.format(k)
		results = {	
			algorithm.__name__: {
				metric : { filename : [] } for metric in metrics
			}
		}

		for sample in range(sampling_size):
			instance = algorithm(n_clusters=k)
			clf = algorithm(n_clusters=k)
			instance.fit(self.dataset)

			for metric in metrics:
				value = Metrics.clustering_evaluation(
					metric, data=self.dataset, centroids=instance.centroids, clf=clf)

				results[algorithm.__name__][metric][filename].append(value)

		return results

	@prepare_directory('results')
	@iterate('algorithms', 'k_range')
	@bargraph('results')
	def feature_importance(self, algorithm, k):
		feature_range = range(self.dataset.shape[1])
		plot_id = '{}_{}'.format(algorithm.__name__, k)
		
		importances = []

		for i in range(30):
			instance = algorithm(n_clusters=k)
			instance.fit(self.dataset)

			new_dataset = self.full_dataset.copy()
			for index in range(new_dataset.shape[0]):
				unlabeled_data = new_dataset[index,:-1]
				new_dataset[index,-1] = instance.predict(unlabeled_data)

			classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
			classifier.fit(new_dataset[:,:-1], new_dataset[:,-1])
			importances.append(classifier.feature_importances_[:])

		importances = np.array(importances)
		mean = map(lambda x: importances[:,x].mean(), feature_range)
		std = map(lambda x: importances[:,x].std(), feature_range)

		return plot_id, self.source.labels, feature_range, mean, std

	@clean('results')
	@prepare_directory('results')
	@bargraph('results')
	def supervised_feature_importance(self):
		feature_range = range(self.dataset.shape[1])
		plot_id = '{}'.format('random_forest')

		importances = []

		for i in range(30):
			classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
			classifier.fit(self.full_dataset[:,:-1], self.full_dataset[:,-1])
			importances.append(classifier.feature_importances_[:])
		importances = np.array(importances)
		mean = map(lambda x: importances[:,x].mean(), feature_range)
		std = map(lambda x: importances[:,x].std(), feature_range)

		return plot_id, self.source.labels, feature_range, mean, std

	# def custom_scatter(self, clusters, k, algorithm):
	# 	dataset = []
	# 	assignments = []

	# 	for cluster in clusters:
	# 		if cluster.shape[0] == 0:
	# 			continue

	# 		classes = cluster[:,-1]
	# 		values, counts = np.unique(classes, return_counts=True)
	# 		max_index = np.argmax(counts)
	# 		most_frequent_class = str(int(values[max_index]))

	# 		for data in cluster:
	# 			dataset.append(data)
	# 			current_class = str(int(data[-1]))
				
	# 			if current_class == '100' and most_frequent_class == '100':
	# 				assignments.append('#00FF00')
	# 			elif current_class == '100' and most_frequent_class == '10':
	# 				assignments.append('b')
	# 			elif current_class == '100' and most_frequent_class == '1':
	# 				assignments.append('r')
	# 			elif current_class == '10' and most_frequent_class == '100':
	# 				assignments.append('c')
	# 			elif current_class == '10' and most_frequent_class == '10':
	# 				assignments.append('#008800')
	# 			elif current_class == '10' and most_frequent_class == '1':
	# 				assignments.append('m')
	# 			elif current_class == '1' and most_frequent_class == '100':
	# 				assignments.append('y')
	# 			elif current_class == '1' and most_frequent_class == '10':
	# 				assignments.append('k')
	# 			elif current_class == '1' and most_frequent_class == '1':
	# 				assignments.append('#005500')

	# 	self.visualize_pca(dataset, assignments, k, algorithm, 3)

	# @clean('results')
	# @iterate('algorithms', 'k_range')
	# def find_errors(self, algorithm, k):
	# 	instance = algorithm(n_clusters=k)
	# 	instance.fit(self.dataset)

	# 	clusters = [[] for _ in range(k)]
	# 	for data in self.full_dataset:
	# 		unclassified_data = data[:-1]
	# 		predicted = instance.predict(unclassified_data)

	# 		clusters[predicted].append(data)

	# 	clusters = map(lambda x: np.array(x), clusters)
	# 	self.custom_scatter(clusters, k, algorithm)

	@prepare_directory('results')
	@iterate('algorithms', 'k_range')
	@log_to_csv('results')
	def find_errors(self, algorithm, k, classifier):
		resume = []
		classifier_name = classifier.__class__.__name__
		file_id = '{}_{}_{}'.format(classifier_name, algorithm.__name__, k)
		labels = self.source.labels + ['Unsupervised', 'Supervised']
		_, _, errors = self.__confusion_labels_kernel(algorithm, k)

		for sample, predicted_label in errors:
			output = classifier.predict([sample[:-1]])[0]
			resume.append(sample.tolist() + [predicted_label, output])

		return file_id, labels, resume

	def find_statistical_errors(self, algorithm, k, classifier):
		sample_size = 50
		all_agreed = np.zeros(sample_size)
		machine_agreed = np.zeros(sample_size)
		all_disagreed = np.zeros(sample_size)
		supervised_and_doctor_agreed = np.zeros(sample_size)

		for i in range(sample_size):
			cases = [0, 0, 0, 0]
			_, right, wrong = self.__confusion_labels_kernel(algorithm, k)

			for sample, predicted_label in right:
				output = classifier.predict([sample[:-1]])[0]

				if int(output) == int(predicted_label):
					cases[0] += 1

			for sample, predicted_label in wrong:
				output = classifier.predict([sample[:-1]])[0]

				if int(output) == int(predicted_label):
					cases[1] += 1

				if int(output) != int(predicted_label) and int(output) != int(sample[-1]):
					cases[2] += 1
				
				if int(output) != int(predicted_label) and int(output) == int(sample[-1]):
					cases[3] += 1



			all_agreed[i] = cases[0]
			machine_agreed[i] = cases[1]
			all_disagreed[i] = cases[2]
			supervised_and_doctor_agreed[i] = cases[3]

		with open('results.txt', 'w+') as f:
			f.write(str(all_agreed.mean()) + ',' + str(all_agreed.std()) + '\n')
			f.write(str(machine_agreed.mean()) + ',' + str(machine_agreed.std()) + '\n')
			f.write(str(all_disagreed.mean()) + ',' + str(all_disagreed.std()) + '\n')
			f.write(str(supervised_and_doctor_agreed.mean()) + ',' + str(supervised_and_doctor_agreed.std()) + '\n')



