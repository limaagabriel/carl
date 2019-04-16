import requests
import numpy as np
import pandas as pd

class Collection(object):
	def all_sets(self):
		def recursive_search(base):
			sets = []
			sets.append(base)

			subsets = base.__subclasses__()
			if len(subsets) == 0:
				return sets
			else:
				for subset in subsets:
					found_elements = recursive_search(subset)
					for element in found_elements:
						sets.append(element)
				return sets

		return recursive_search(self.base)

class Dataset(object):
	def __init__(self, path=None, delimiter=None):
		method = self.__download if path is None else self.__import_dataset
		delimiter = self.delimiter if delimiter is None else delimiter
		path = self.url if path is None else path

		print('Loaded {}'.format(self.__class__.__name__))
		self.dataset = self.__process_features(method(path, delimiter))
		self.unlabeled_data = self.__remove_classes(self.dataset)
		self.feature_shape = self.unlabeled_data.shape
		self.classes = self.__classes()

	def split(self, proportion):
		dataframe = pd.DataFrame(data=self.dataset)
		training = dataframe.sample(frac=proportion)
		test = dataframe[~dataframe.index.isin(training.index)]

		training = training.values
		test = test.values

		return training[:,0:-1], training[:,-1], test[:,0:-1], test[:,-1]

	def __process_features(self, dataset):
		pipeline = [self.__select_features,
					self.__scale_features]

		for function in pipeline:
			dataset = function(dataset)

		return dataset

	def __select_features(self, dataset):
		try:
			self.labels = list(map(self.labels.__getitem__, self.features))
			return dataset[:, self.features]		
		except AttributeError:
			return dataset

	def __scale_features(self, dataset):
		try:
			scaler = self.scaler()
			data = dataset[:,0:-1]
			scaled_data = scaler.fit_transform(data)
			output = np.zeros(dataset.shape)

			output[:,0:-1] = scaled_data
			output[:,-1] = dataset[:,-1]

			return output
		except AttributeError:
			return dataset

	def __remove_classes(self, dataset):
		return dataset[:,:-1]

	def __import_dataset(self, path, delimiter=','):
		return np.genfromtxt(path, delimiter=delimiter)

	def __download(self, url, delimiter=','):
		response = requests.get(url)
		return self.__string_to_dataset(response.text, delimiter)

	def __classes(self):
		all_classes = self.dataset[:,-1]
		unique_classes = set(all_classes.tolist())
		return sorted(unique_classes)

	def __string_to_dataset(self, text, delimiter):
		lines = text.splitlines()
		to_float = lambda y: float(y)
		tokenizer = lambda x: map(to_float, x.split(delimiter))
		entities = list(map(tokenizer, lines))
		return np.array(entities)



