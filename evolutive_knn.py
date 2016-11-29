from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import copy
from individual import Individual
import random

class EvolutiveKNN:
    """Implementation of an evolutive version of KNN.
    This class finds the best K and the best weigths for a given training set
    """
    """
    EvolutiveKNN initializer

    Parameters:
        training_examples: Array of features, each feature is an array of floats
            example: [[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]]
        training_labels: Array of integers that are the labels for each feature.
            example: [0, 1, 0]
        Observation: the first label is the class of the first feature, and so on.
    
    Usage:
        classifier = EvolutiveKNN([[1, 2, 3, 1], [1, 4, 2, 8], [1, 1, 2, 1]], [0, 1, 0])
    """
    def __init__(self, training_examples, training_labels, ts_size = 0.5):
        test_size = int(ts_size * len(training_labels))
        self._features_size = len(training_examples[0])
        self._create_test(
            np.array(training_examples), np.array(training_labels), test_size
        )

    """This method is responsible for training the evolutive KNN based on the
    given parameters

    Parameters:
        population_size: The size of the population.
        mutation_rate: Chance of occuring a mutation on a individual.
        max_generations: Stopping criteria, maximum number of generations.
        max_accuracy: Stopping criteria, if an idividual have an accuracy bigger than max_accuracy the execution stops.
        max_k: Maximum number of neighbors, if no max_k is provided the maximum possible is used.
        max_neighbor_weight: Maximum possible weight for neighbors.
        max_feature_weight: Maximum possible weight for features.
        elitism_rate: Elitism rate, percentage of best individuals that will be passed to another generation.
        tournament_size: The percentage of the non-elite population that will be selected at each tournament.
    """
    def train(self, population_size=50, mutation_rate=0.02, max_generations=50, max_accuracy=1.0, max_k=None, max_neighbors_weight=10, max_features_weight=10, elitism_rate=0.0, tournament_size=0.25):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_accuracy = max_accuracy
        self.max_k = max_k
        self.max_neighbors_weight = max_neighbors_weight
        self.max_features_weight = max_features_weight
        self.elitism_rate = elitism_rate
        self.elitism_real_value = int(self.elitism_rate * self.population_size)
        self.tournament_size = tournament_size
        self.global_best = Individual(1, [1], [1] * self._features_size)
        self.hall_of_fame = []
        self.best_of_each_generation = []
        self._train()

    def _train(self):
        population = self._start_population()
        generations = 0
        # print generations
        self._calculate_fitness_of_population(population, generations)
        while not self._should_stop(generations):
            generations += 1
            # print generations
            population = self._create_new_population(population)
            self._calculate_fitness_of_population(population, generations)

    def _should_stop(self, generations):
        best_fitness = self.global_best.fitness
        if self.max_generations < generations or best_fitness >= self.max_accuracy:
            return True
        return False

    def _create_new_population(self, old_population):
        sorted_old_population = sorted(
            old_population,
            key=lambda individual: individual.fitness,
            reverse=True
        )
        elite = self._get_elite(sorted_old_population)
        non_elite = self._get_non_elite(sorted_old_population)
        new_population = elite
        while len(new_population) < self.population_size:
            new_population.append(
                self._generate_child(non_elite)
            )
        return new_population

    def _generate_child(self, population):
        parent1 = self._tournament(population)
        parent2 = self._tournament(population)
        kid = self._crossover(parent1, parent2)
        # print 'CROSSOVER'
        # print parent1.k, parent1.neighbors_weights, parent1.features_weights
        # print '-------------'
        # print parent2.k, parent2.neighbors_weights, parent2.features_weights
        # print '--------------'
        # print kid.k, kid.neighbors_weights, kid.features_weights
        # print '=--------------='
        return kid

    def _tournament(self, population):
        number_of_individuals = int(len(population) * self.tournament_size)
        selected = random.sample(
            xrange(len(population)), number_of_individuals
        )
        best = sorted(selected)[0]
        return population[best]

    def _crossover(self, parent1, parent2):
        k = self._random_between(parent1.k, parent2.k)
        return Individual(
            k,
            self._neighbors_crossover(k, parent1, parent2),
            self._features_crossover(parent1, parent2))

    def _features_crossover(self, parent1, parent2):
        colaboration1 = int(self._features_size/2.0)
        colaboration2 = int(self._features_size/2.0)
        weights_p1 = parent1.features_weights[:colaboration1]
        weights_p2 = parent2.features_weights[colaboration2:]
        weights = weights_p1 + weights_p2
        mutate = random.uniform(0, 1)
        if mutate < self.mutation_rate:
            weights = self._mutate_weights(weights, self.max_features_weight)
        return weights

    def _neighbors_crossover(self, k, parent1, parent2):
        k1 = parent1.k
        k2 = parent2.k
        colaboration1 = int(np.floor(k * (k1/float(k1 + k2))))
        colaboration2 = int(np.ceil(k * (k2/float(k1 + k2))))
        weights_p1 = random.sample(parent1.neighbors_weights, colaboration1)
        weights_p2 = random.sample(parent2.neighbors_weights, colaboration2)
        weights = weights_p1 + weights_p2
        mutate = random.uniform(0, 1)
        if mutate < self.mutation_rate:
            weights = self._mutate_weights(weights, self.max_neighbors_weight)
        return weights

    def _random_between(self, number1, number2):
        if random.randint(0, 1) == 0:
            result = number1
        else:
            result = number2
        return result

    def _mutate_weights(self, weights, maximum):
        mutated = weights
        index = random.randint(0, len(weights) - 1)
        mutated[index] = random.randint(0, maximum)
        return mutated

    def _get_elite(self, population):
        return population[:self.elitism_real_value]

    def _get_non_elite(self, population):
        return population[self.elitism_real_value:]

    def _start_population(self):
        max_k = self.max_k
        if max_k is None: max_k = len(self.training_labels)
        population = []
        for _ in xrange(self.population_size):
            k = random.randint(1, max_k)
            neighbors_weights = [
                random.choice(range(self.max_neighbors_weight)) for _ in xrange(k)
            ]
            features_weights = [
                random.choice(range(self.max_features_weight)) for _ in xrange(self._features_size)
            ]
            population.append(Individual(k, neighbors_weights, features_weights))
        return population

    def _calculate_fitness_of_population(self, population, generation):
        population_best = Individual(1, [1], [1] * self._features_size)
        for element in population:
            self._calculate_fitness_of_individual(element)
            if population_best.fitness < element.fitness:
                population_best = copy.deepcopy(element)
        self.best_of_each_generation.append(population_best)
        if self.global_best.fitness < population_best.fitness:
                self._change_global_best(population_best, generation)

    def _change_global_best(self, element, generation):
        self.hall_of_fame.append(
            {'individual': element, 'generation': generation}
        )
        self.global_best = element

    def _calculate_fitness_of_individual(self, element):

        def _element_weights(distances):
            return element.neighbors_weights

        w = element.features_weights
        kneigh = KNeighborsClassifier(n_neighbors=element.k, weights=_element_weights)
        try:
            kneigh.fit(self.training_examples * w, self.training_labels)
        except:
            print element.k
            print element.neighbors_weights
            print element.features_weights
        element.fitness = kneigh.score(
            self.test_examples * w, self.test_labels
        )
        # print '-------'
        # print element.k
        # print element.weights
        # print element.fitness
        # print '-------'

    def _create_test(self, tr_examples, tr_labels, test_size):
        self.training_examples = []
        self.training_labels = []
        self.test_examples = []
        self.test_labels = []

        test_indexes = random.sample(xrange(len(tr_labels)), test_size)

        self.test_examples = tr_examples[test_indexes]
        self.test_labels = tr_labels[test_indexes]
        for index in xrange(len(tr_labels)):
            if index not in test_indexes:
                self.training_examples.append(tr_examples[index])
                self.training_labels.append(tr_labels[index])
        self.training_examples = np.array(self.training_examples)
        self.training_labels = np.array(self.training_labels)