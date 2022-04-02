import keras_genetic


class CompletelyRandomBreeder(keras_genetic.Breeder):
    """CompletelyRandomBreeder creates fully random weight sets."""

    def __init__(self, parents_per_generation, keep_parents=True):
        self.parents_per_generation = parents_per_generation
        self.keep_parents = keep_parents

    def offspring(self):
        mother = self._parents[0]
        return self.fully_random_weight_set(mother)

    def update_state(self, generation):
        self._parents = generation[:parents_per_generation]

    def population(self, population_size):
        result = []
        for _ in range(population_size):
            result.append(self.offspring())
        if self.keep_parents:
            result = result + self._parents
        return result
