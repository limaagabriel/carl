from experiments.researcher import Researcher
from data.huoc.datasets import HuocCollection
from clustering.swarm.pso.PSOC import PSOC
from clustering.swarm.pso.PSOCKM import PSOCKM
from clustering.swarm.pso.KMPSOC import KMPSOC
from clustering.traditional.KMeans import KMeans
from clustering.traditional.FCMeans import FCMeans

min_k = 3
max_k = 15
sampling_size = 30
k_range = range(min_k, max_k + 1)
sources = HuocCollection().all_sets()
algorithms = [FCMeans]

metrics = ['gap', 'calinskiHarabaszIndex', 'hartiganIndex', 'silhouette', 'daviesBouldin', 'dunnIndex']

for source in sources:
	researcher = Researcher(source())
	researcher.find_k(k_range=k_range,
					  algorithms=algorithms,
					  sampling_size=sampling_size,
					  metrics=metrics)


