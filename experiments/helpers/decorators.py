import os
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from functools import wraps
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def iterate(*args):
	def mapper(fn):
		@wraps(fn)
		def wrapper(self, **kwargs):
			iterables = map(lambda x: kwargs[x], args)
			map_set = itertools.product(*iterables)
			total = reduce(lambda a, b: a * len(b), iterables, 1)

			for arg in args:
				kwargs.pop(arg)

			try:
				for data in tqdm(map_set, total=total):
					fn(self, *data, **kwargs)
			except KeyboardInterrupt:
				pass

		return wrapper
	return mapper


def bargraph(base):
	def save_statistical(path, y, std):
		with open(path + '.csv', 'w+') as file:
			file.write(','.join(map(lambda x: str(x), y)))
			file.write('\n')
			file.write(','.join(map(lambda x: str(x), std)))


	def plot_bargraph(path, labels, plot_id, x, y, std):
		path = os.path.join(path, plot_id)

		plt.bar(x, height=y, yerr=std)
		plt.xticks(x, labels, rotation=60)
		plt.subplots_adjust(bottom=0.3)
		plt.savefig('{}.png'.format(path))
		plt.clf()
		plt.close()
		
		save_statistical(path, y, std)

	def bargrapher(fn):
		@wraps(fn)
		def wrapper(self, *args, **kwargs):
			fn_name = fn.__name__
			source_name = self.source.__class__.__name__
			path = os.path.join(base, source_name, fn_name)
			plot_bargraph(path, self.source.labels, *fn(self, *args, **kwargs))

		return wrapper
	return bargrapher

def errorbar(x_axis, mean_index, std_index):
	def plotter(fn):
		@wraps(fn)
		def wrapper(self, *args, **kwargs):
			source_name = self.source.__class__.__name__
			base_path = os.path.join('results', source_name, fn.__name__)
			fn(self, *args, **kwargs)
			for root, dirs, files in os.walk(base_path):
				if len(files) > 0:
					ks = []
					mean = []
					std_dev = []

					for x in kwargs[x_axis]:
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

		return wrapper
	return plotter

def confusion_matrix(fn):
	def convert(matrix, class_names):
		keys = matrix.keys()
		names = map(lambda x: class_names[x], keys)
		mat = map(lambda x: np.array(map(lambda y: matrix[x][y], keys)), keys) 

		return (np.array(mat), np.array(names))


	def plot(plot_id, source_name, cm, classes):
		cmap = plt.cm.Blues
		filename = '{}.png'.format(plot_id)
		filepath = os.path.join('results', source_name, fn.__name__, filename)

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes, rotation=45)

		fmt = 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()
		plt.savefig(filepath)
		plt.clf()
		plt.close()


	@wraps(fn)
	def wrapper(self, *args, **kwargs):
		class_names = kwargs.pop('class_names')
		source_name = self.source.__class__.__name__

		plot_id, matrix = fn(self, *args, **kwargs)
		plot(plot_id, source_name, *convert(matrix, class_names))

	return wrapper

def scatter_plot(fn):
	def plot(source_name, plot_id, data, assignments, names=None):
		cmap = plt.cm.get_cmap('viridis')
		filename = '{}.png'.format(plot_id)
		filepath = os.path.join('results', source_name, fn.__name__, filename)

		if data.shape[1] == 2:
			x = data[:,0]
			y = data[:,1]
			plt.scatter(x, y, c=assignments, cmap=cmap)

			if names is not None:
				plt.xlabel(names[0])
				plt.ylabel(names[1])

		else:
			x = data[:,0]
			y = data[:,1]
			z = data[:,2]
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(x, y, z, c=assignments, cmap=cmap)

			if names is not None:
				ax.set_xlabel(names[0])
				ax.set_ylabel(names[1])
				ax.set_zlabel(names[2])

		plt.savefig(filepath)
		plt.clf()
		plt.close()



	@wraps(fn)
	def wrapper(self, *args, **kwargs):
		source_name = self.source.__class__.__name__
		plot_data = fn(self, *args, **kwargs)

		plot(source_name, *plot_data)

	return wrapper

def prepare_directory(base):
	if not os.path.exists(base):
		os.mkdir(base)
	
	def organizer(fn):
		@wraps(fn)
		def wrapper(self, *args, **kwargs):
			source_name = self.source.__class__.__name__
			root = os.path.join(base, source_name, fn.__name__)
			if not os.path.exists(root):
				os.makedirs(root)

			fn(self, *args, **kwargs)
		return wrapper
	return organizer

def clean(base):
	def cleaner(fn):
		@wraps(fn)
		def wrapper(self, *args, **kwargs):
			source_name = self.source.__class__.__name__
			root = os.path.join(base, source_name, fn.__name__)
			if os.path.exists(root):
				shutil.rmtree(root)

			fn(self, *args, **kwargs)

		return wrapper
	return cleaner

def save_results_with_statistical_data(fn):
	def depth(d):
		if isinstance(d, dict):
			return 1 + (max(map(depth, d.values())) if d else 0)
		return 0			

	def connect(base, paths, result):
		def finder(path):
			base_path = os.path.relpath(path, base)
			tokens = base_path.split(os.sep)

			data = result
			for token in tokens:
				data = data[token]

			return (path, data)

		return map(finder, paths)

	def extract_paths(fn, result, source_path):
		base = os.path.join('results', source_path, fn.__name__)

		def recursive(current, paths):
			if isinstance(current, dict):
				p = []

				for key in current.keys():
					new_paths = map(lambda x: os.path.join(x, key), paths)
					p = p + recursive(current[key], new_paths)
				return p
			else:
				return paths

		return connect(base, recursive(result, [base]), result)

	@wraps(fn)
	def wrapper(self, *args, **kwargs):
		result = fn(self, *args, **kwargs)
		source_name = self.source.__class__.__name__
		paths = extract_paths(fn, result, source_name)

		for path, data in paths:
			base = os.path.split(path)[0]
			if not os.path.exists(base):
				os.makedirs(base)
			with open(path, 'w+') as file:
				mat = np.array(data)
				mean = np.mean(mat)
				std_dev = np.std(mat)

				data.insert(0, std_dev)
				data.insert(0, mean)

				file.write(','.join(map(lambda x: str(x), data)))

	return wrapper
