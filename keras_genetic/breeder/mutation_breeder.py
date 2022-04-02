import random

import numpy as np

from keras_genetic import core
from keras_genetic.breeder.breeder import Breeder


class MutationBreeder(Breeder):
    """MutationBreeder randomly mutates features on individuals."""

    def __init__(
        self,
        parents_per_generation,
        keep_probability=0.9,
        keep_parents=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_probability = keep_probability
        self.parents_per_generation = parents_per_generation
        self.keep_parents = keep_parents
        self.parents = None

    def update_state(self, generation):
        self._parents = generation[: self.parents_per_generation]

    def offspring(self):
        parents = self._parents
        if not parents:
            raise RuntimeError(
                "`MutationBreeder.offspring()` called before "
                "`update_state()`.  Please call `update_state()` at least once before "
                "calling `offspring()`."
            )
        mother = random.choice(parents)
        offspring_weights = []

        # each 'mother.weight', 'father.weight' we have a vector of weights
        # we must traverse the entire array and sample a random one for each
        for w in mother.weights:
            shape = w.shape
            w = w.flatten()

            result = []
            for i in range(w.shape[0]):
                if random.uniform(0, 1) > self.keep_probability:
                    result.append(self.initializer())
                    continue
                result.append(w[i])

            offspring_weights.append(np.array(result).reshape(shape))

        return core.Individual(offspring_weights, model=mother.model)

    def population(self, population_size):
        result = []
        for _ in range(population_size):
            result.append(self.offspring())
        if self.keep_parents:
            result = result + self._parents
        return result
