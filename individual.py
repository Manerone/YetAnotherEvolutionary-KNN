class Individual:
    def __init__(self, k, neighbors_weights, features_weights, fitness=0.0):
        if k != len(neighbors_weights):
            raise Exception('K and neighbors_weights size are diferent')
        self.k = k
        self.neighbors_weights = neighbors_weights
        self.features_weights = features_weights
        self.fitness = fitness

    def set_fitness(self, fitness):
        self.fitness = fitness