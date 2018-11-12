import os

metrics = ['hue', 'hua', 'top', 'maravilha']
k = 2
x = {	'sei': {
	
		'ok': {
			metric : { str(k) : ['ok'] } for metric in metrics
		}
}
	}


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

	def extract_paths(fn, result):
		base = os.path.join('results', fn.__name__)

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

	def wrapper(self, **kwargs):
		result = fn(self, **kwargs)
		paths = extract_paths(fn, result)

		for path, data in paths:
			with open(path, 'w+') as file:
				mat = np.array(data)
				mean = np.mean(mat)
				std_dev = np.std(mat)

				data.insert(0, std_dev)
				data.insert(0, mean)

				file.write(data)

	return wrapper

@save_results_with_statistical_data
def op(x):
	return x

print(op(x))