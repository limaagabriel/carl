from experiments.researcher import Researcher
from data.huoc.datasets import HuocCollection
from clustering.traditional.FCMeans import FCMeans
from sklearn.ensemble import RandomForestClassifier

min_k = 3
max_k = 15
k_range = range(min_k, max_k + 1)
sources = HuocCollection().all_sets()
algorithms = [FCMeans]

for source in sources:
	s = source()
	classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
	ax, ay, bx, by = s.split(0.7)


	classifier.fit(ax, ay)
	print(classifier.score(bx, by))
	break

	researcher = Researcher(s)
	researcher.find_errors(k_range=k_range,
					  	   algorithms=algorithms,
					  	   classifier=classifier)


