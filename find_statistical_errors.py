from experiments.researcher import Researcher
from data.huoc.datasets import CognitiveTestsDataset
from clustering.traditional.FCMeans import FCMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

min_k = 13
max_k = 13
k_range = range(min_k, max_k + 1)
sources = [CognitiveTestsDataset]
algorithms = [FCMeans]

for source in sources:
	s = source()

	while True:
		hidden_layers = (s.feature_shape[1] * 2 + 1,)
		classifier = MLPClassifier(learning_rate='adaptive',
							   learning_rate_init=0.1,
							   activation='logistic',
							   early_stopping=True,
							   batch_size=1,
							   hidden_layer_sizes=hidden_layers)

		ax, ay, bx, by = s.split(0.7)

		classifier.fit(ax, ay)
		score = classifier.score(bx, by)
		print(score)
		if score > 0.85:
			break

	researcher = Researcher(s)
	researcher.find_statistical_errors(k=13,
					  	   algorithm=FCMeans,
					  	   classifier=classifier)


