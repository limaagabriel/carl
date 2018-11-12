from experiments.researcher import Researcher
from clustering.traditional.KMeans import KMeans
from clustering.traditional.FCMeans import FCMeans
from data.huoc.datasets import HuocCollection

min_k = 3
max_k = 15
sources = HuocCollection().all_sets()
k_range = range(min_k, max_k + 1)
algorithms = [KMeans, FCMeans]

class_names = {
	'1': 'Dementia',
	'10': 'MCI',
	'100': 'Control'
}

for source in sources:
	researcher = Researcher(source())
	researcher.confusion_labels(k_range=k_range,
								algorithms=algorithms,
								class_names=class_names)
