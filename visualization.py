from data.huoc.datasets import HuocDataset
from experiments.researcher import Researcher
from clustering.traditional.FCMeans import FCMeans


min_k = 2
max_k = 12
source = HuocDataset()
k_range = range(min_k, max_k + 1)
algorithms = [FCMeans]

researcher = Researcher(source)
researcher.visualize(k_range=k_range,
					 algorithms=algorithms)
