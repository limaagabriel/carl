from experiment.researcher import Researcher
from data.input import HuocDataset, CognitiveTestsDataset

from clustering.swarm.pso.PSOC import PSOC
from clustering.swarm.pso.PSOCKM import PSOCKM
from clustering.swarm.pso.KMPSOC import KMPSOC
from clustering.traditional.KMeans import KMeans
from clustering.traditional.FCMeans import FCMeans

min_k = 2
max_k = 15
sampling_size = 40
k_range = range(min_k, max_k + 1)
sources = [HuocDataset, CognitiveTestsDataset]
algorithms = [KMeans, FCMeans, KMPSOC, PSOC, PSOCKM]

for source in sources:
	researcher = Researcher(source())
	researcher.explore_k(k_range=k_range,
						 metrics=metrics,
						 algorithms=algorithms,
						 sampling_size=sampling_size)


