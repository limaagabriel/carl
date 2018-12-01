from experiments.researcher import Researcher
from data.huoc.datasets import HuocCollection
from clustering.traditional.FCMeans import FCMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

min_k = 3
max_k = 15
k_range = range(min_k, max_k + 1)
sources = HuocCollection().all_sets()
algorithms = [FCMeans]

for source in sources:
	s = source()
	hidden_layers = (s.feature_shape[1] * 2 + 1,)
	classifier = MLPClassifier(learning_rate='constant',
							   learning_rate_init=0.1,
							   activation='logistic',
							   early_stopping=True,
							   hidden_layer_sizes=hidden_layers)

	ax, ay, bx, by = s.split(0.7)

	classifier.fit(ax, ay)
	print(classifier.score(bx, by))

	researcher = Researcher(s)
	researcher.find_errors(k_range=k_range,
					  	   algorithms=algorithms,
					  	   classifier=classifier)


