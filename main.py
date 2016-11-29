from evolutive_knn import EvolutiveKNN
from banknote_loader import BanknoteLoader


banknote = BanknoteLoader('./datasets/banknote.data')
classifier = EvolutiveKNN(banknote.examples, banknote.labels)
classifier.train()
print classifier.global_best.k
print classifier.global_best.neighbors_weights
print classifier.global_best.features_weights
print '-------------------'
print map(lambda x: [x['individual'].fitness, x['generation']], classifier.hall_of_fame)
print map(lambda x: x.fitness, classifier.best_of_each_generation)
