from experiments.researcher import Researcher
from data.huoc.datasets import HuocCollection

sources = HuocCollection().all_sets()

for source in sources:
	researcher = Researcher(source())
	researcher.supervised_feature_importance()


